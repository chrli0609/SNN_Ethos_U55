# Modular Serial Port Reader with CSV Export
# This script reads data from a serial port, parses multiple configurable patterns, and exports each to separate CSV files

param (
    [string]$fileName = "neuron_data_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [string]$csvDir = "csv/",
    [string]$portName = "COM7",
    [int]$baudRate = 115200,
    [int]$DataBits = 8
)

# CONFIGURATION: Define patterns to look for
# Each pattern should have: Name, Regex, FilenameSuffix, Units
$patterns = @(
    @{
        Name = "Ticks Elapsed"
        Regex = "Ticks elapsed for layer once in it: (\d+) = (\d+)(?:\s*ms)?"
        FilenameSuffix = "_ticks"
        Units = "(ms)"
        Description = "Processing ticks in milliseconds"
    },
    @{
        Name = "NPU Cycles" 
        Regex = "Npu cycles for it: (\d+) = (\d+)"
        FilenameSuffix = "_npu_cycles"
        Units = "(cycles)"
        Description = "NPU processing cycles"
    }
    @{
        Name = "NPU MAC Active" 
        Regex = "Npu MAC Active for it: (\d+) = (\d+)"
        FilenameSuffix = "_npu_mac"
        Units = "(cycles)"
        Description = "Cycles where the MAC on the NPU is active"
    }
    # ADD NEW PATTERNS HERE - Example:
    # @{
    #     Name = "Memory Usage"
    #     Regex = "Memory used for it: (\d+) = (\d+)(?:\s*KB)?"
    #     FilenameSuffix = "_memory"
    #     Units = "(KB)"
    #     Description = "Memory usage in kilobytes"
    # }
)

# Data structure to hold all pattern data
$allPatternData = @{}
$patternMaxIterations = @{}

# Initialize data structures for each pattern
foreach ($pattern in $patterns) {
    $allPatternData[$pattern.Name] = @{}
    $patternMaxIterations[$pattern.Name] = 0
}

# Configure and open the serial port
$port = New-Object System.IO.Ports.SerialPort $portName,$baudRate,None,$DataBits,one
$port.Encoding = [System.Text.Encoding]::UTF8
$port.Open()

# Data storage
$currentNeurons = 0
$rawOutput = ""

Write-Host "Reading from $portName. Press 'q' and Enter to stop reading..."
Write-Host "Configured to capture $($patterns.Count) different data types:"
foreach ($pattern in $patterns) {
    Write-Host "  - $($pattern.Name): $($pattern.Description)"
}
Write-Host ""

# Read from the port until user presses 'q'
while ($port.IsOpen) {
    if ($port.BytesToRead -gt 0) {
        $data = $port.ReadExisting()
        $rawOutput += $data
        Write-Host $data -NoNewline
    }
    
    # Check if user pressed 'q' to exit
    if ([console]::KeyAvailable) {
        $key = [console]::ReadKey($true).Key
        if ($key -eq "Q") {
            break
        }
    }
    Start-Sleep -Milliseconds 100
}

$port.Close()
Write-Host "Port closed. Processing data..."

# Process the captured data
$lines = $rawOutput -split "`r`n"

foreach ($line in $lines) {
    # Check for neuron count line
    if ($line -match "Num neurons = (\d+)") {
        $currentNeurons = [int]$matches[1]
        
        # Initialize storage for this neuron count across all patterns
        foreach ($pattern in $patterns) {
            if (-not $allPatternData[$pattern.Name].ContainsKey($currentNeurons)) {
                $allPatternData[$pattern.Name][$currentNeurons] = @{}
            }
        }
    }
    # Check each configured pattern
    else {
        foreach ($pattern in $patterns) {
            if ($line -match $pattern.Regex) {
                $iteration = [int]$matches[1]
                $value = [int]$matches[2]
                
                if ($currentNeurons -gt 0) {
                    $allPatternData[$pattern.Name][$currentNeurons][$iteration] = $value
                    $patternMaxIterations[$pattern.Name] = [Math]::Max($patternMaxIterations[$pattern.Name], $iteration)
                }
                break  # Exit pattern loop once we find a match
            }
        }
    }
}

# Function to create and save CSV for a given pattern
function Save-PatternToCSV {
    param(
        [hashtable]$PatternData,
        [string]$PatternName,
        [string]$Units,
        [string]$FilenameSuffix,
        [int]$MaxIteration,
        [string]$BaseFileName,
        [string]$CsvDirectory
    )
    
    $neuronCounts = $PatternData.Keys | Sort-Object
    if ($neuronCounts.Count -eq 0) {
        Write-Host "No data found for pattern: $PatternName"
        return $null
    }
    
    # Create CSV content
    $csv = New-Object System.Text.StringBuilder
    $csvFileName = Join-Path $CsvDirectory $BaseFileName
    
    # Build header row
    $headerRow = "Iteration"
    foreach ($neurons in $neuronCounts) {
        $headerRow += ",neurons_${neurons}_${Units}"
    }
    [void]$csv.AppendLine($headerRow)
    
    # Build data rows
    for ($i = 0; $i -le $MaxIteration; $i++) {
        $row = "$i"
        foreach ($neurons in $neuronCounts) {
            if ($PatternData[$neurons].ContainsKey($i)) {
                $row += ",$($PatternData[$neurons][$i])"
            } else {
                $row += ","  # Empty value if this iteration doesn't exist for this neuron count
            }
        }
        [void]$csv.AppendLine($row)
    }
    
    # Save to CSV file
    $csv.ToString() | Out-File -Encoding utf8 $csvFileName
    return $csvFileName
}

# Create output directory structure
$outputDir = Join-Path $csvDir $fileName
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    Write-Host "Created output directory: $outputDir"
}

# Process and save each pattern's data
$savedFiles = @()
foreach ($pattern in $patterns) {
    $patternData = $allPatternData[$pattern.Name]
    $maxIteration = $patternMaxIterations[$pattern.Name]
    
    if ($maxIteration -gt 0) {
        # Use just the suffix as filename in the dedicated directory
        $patternFileName = $pattern.FilenameSuffix.TrimStart('_') + ".csv"
        $savedFile = Save-PatternToCSV -PatternData $patternData -PatternName $pattern.Name -Units $pattern.Units -FilenameSuffix $pattern.FilenameSuffix -MaxIteration $maxIteration -BaseFileName $patternFileName -CsvDirectory $outputDir
        
        if ($savedFile) {
            $savedFiles += $savedFile
            Write-Host "$($pattern.Name) data saved to: $savedFile"
            
            # Display summary for this pattern
            $neuronCounts = $patternData.Keys | Sort-Object
            Write-Host "  - Neuron counts: $($neuronCounts -join ', ')"
            Write-Host "  - Max iterations: $maxIteration"
            Write-Host "  - Total data points: $(($patternData.Values | ForEach-Object { $_.Count } | Measure-Object -Sum).Sum)"
        }
    } else {
        Write-Host "No data captured for pattern: $($pattern.Name)"
    }
    Write-Host ""
}

# Overall summary
Write-Host "=== PROCESSING COMPLETE ==="
Write-Host "Total files created: $($savedFiles.Count)"
Write-Host "Files created:"
foreach ($file in $savedFiles) {
    Write-Host "  - $file"
}

# Plot all created CSV files
if ($savedFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "Generating plots..."
    foreach ($file in $savedFiles) {
        & python plot_csv.py $file
    }
}