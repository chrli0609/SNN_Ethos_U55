# Modular Serial Port Reader with CSV Export
# This script reads data from a serial port, parses multiple configurable patterns, and exports each to separate CSV files
# Modified to use input neuron sizes as rows instead of iterations

param (
    [string]$npuConfig = "neuron_data_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [string]$csvDir = "../csv/",
    [string]$portName = "COM7",
    [int]$baudRate = 115200,
    [int]$DataBits = 8
)

# Set console and output encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# CONFIGURATION: Define patterns to look for
# Each pattern should have: Name, Regex, filename, Units
$patterns = @(
    @{
        Name = "CPU Microsecs Elapsed"
        Regex = "Ticks elapsed for layer once in it: (\d+) = (\d+)(?:\s*[mµ]s)?"
        filename = "cpu_microsecs"
        Units = "(us)"
        Description = "Processing ticks in microseconds"
    },
    @{
        Name = "NPU Cycles" 
        Regex = "Npu cycles for it: (\d+) = (\d+)"
        filename = "npu_cycles"
        Units = "(cycles)"
        Description = "NPU processing cycles"
    }
    @{
        Name = "NPU MAC Active" 
        Regex = "Npu MAC Active for it: (\d+) = (\d+)"
        filename = "npu_mac"
        Units = "(operations)"
        Description = "Cycles where the MAC on the NPU is active"
    }
    @{
        Name = "Npu MAC 8 bit Active" 
        Regex = "Npu MAC 8bit Active for it: (\d+) = (\d+)"
        filename = "npu_mac_8bit"
        Units = "(operations)"
        Description = "Number of operations the 8 bit MAC on the NPU is doing"
    }
    @{
        Name = "NPU AXI0 Read" 
        Regex = "Npu AXI0 Reads for it: (\d+) = (\d+)"
        filename = "npu_axi0_reads"
        Units = "(events)"
        Description = "Number of AXI0 Reads Accepted"
    }
    @{
        Name = "NPU BlockDep Stalls" 
        Regex = "Npu Blockdep Stalls for it: (\d+) = (\d+)"
        filename = "npu_blockdep_stalls"
        Units = "(cycles)"
        Description = "Cycles where the MAC is stalled due to Block Dependencies"
    }
    @{
        Name = "NPU CCs where DPU is Active" 
        Regex = "Npu CCs where DPU is Active for it: (\d+) = (\d+)"
        filename = "npu_cc_dpu_active"
        Units = "(cycles)"
        Description = "NPU Cycles where DPUs are active"
    }
    @{
        Name = "Block Configuration" 
        Regex = "Block Configuration for it: (\d+) = (\d+)"
        filename = "npu_block_config"
        Units = "(cycles)"
        Description = "NPU Block Config for each input output size combination"
    }
    # ADD NEW PATTERNS HERE - Example:
    # @{
    #     Name = "Memory Usage"
    #     Regex = "Memory used for it: (\d+) = (\d+)(?:\s*KB)?"
    #     filename = "_memory"
    #     Units = "(KB)"
    #     Description = "Memory usage in kilobytes"
    # }
)

# Data structure to hold all pattern data
# Structure: $allPatternData[PatternName][InputNeurons][OutputNeurons] = @{Value = X, BlockConfig = Y}
$allPatternData = @{}
$allInputNeurons = @()
$allOutputNeurons = @()
$allBlockConfigs = @()

# Initialize data structures for each pattern
foreach ($pattern in $patterns) {
    $allPatternData[$pattern.Name] = @{}
}

# Configure and open the serial port
$port = New-Object System.IO.Ports.SerialPort $portName,$baudRate,None,$DataBits,one
$port.Encoding = [System.Text.Encoding]::UTF8
$port.Open()

# Data storage
$currentInputNeurons = 0
$currentOutputNeurons = 0
$currentBlockConfig = 0
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
Write-Host "Total lines captured: $(($rawOutput -split "`r`n").Count)"

# Process the captured data
$lines = $rawOutput -split "`r`n"

foreach ($line in $lines) {
    # Check for block configuration line
    if ($line -match "Block Configuration = \((\d+),\s*(\d+),\s*(\d+)\)") {
        $currentBlockConfig = [int]$matches[3]  # val3 is the third value
        Write-Host "Found Block Configuration: val3 = $currentBlockConfig"
        
        # Track all block configurations we've seen
        if ($allBlockConfigs -notcontains $currentBlockConfig) {
            $allBlockConfigs += $currentBlockConfig
        }
    }
    # Check for input neuron count line
    elseif ($line -match "Num input neurons = (\d+)") {
        $currentInputNeurons = [int]$matches[1]
        Write-Host "Found Input Neurons: $currentInputNeurons"
        
        # Track all input neuron sizes we've seen
        if ($allInputNeurons -notcontains $currentInputNeurons) {
            $allInputNeurons += $currentInputNeurons
        }
        
        # Initialize storage for this input neuron count across all patterns
        foreach ($pattern in $patterns) {
            if (-not $allPatternData[$pattern.Name].ContainsKey($currentInputNeurons)) {
                $allPatternData[$pattern.Name][$currentInputNeurons] = @{}
            }
        }
    }
    # Check for output neuron count line
    elseif ($line -match "Num output neurons = (\d+)") {
        $currentOutputNeurons = [int]$matches[1]
        Write-Host "Found Output Neurons: $currentOutputNeurons"
        
        # Track all output neuron sizes we've seen
        if ($allOutputNeurons -notcontains $currentOutputNeurons) {
            $allOutputNeurons += $currentOutputNeurons
        }
    }
    # Check each configured pattern
    else {
        foreach ($pattern in $patterns) {
            # Modified regex to expect iteration 0 (since there's only one iteration now)
            if ($line -match $pattern.Regex) {
                $iteration = [int]$matches[1]
                $value = [int]$matches[2]
                
                Write-Host "Found $($pattern.Name): iteration=$iteration, value=$value (Input: $currentInputNeurons, Output: $currentOutputNeurons)"
                
                # Only process if we have valid neuron counts and this is iteration 0
                if ($currentInputNeurons -gt 0 -and $currentOutputNeurons -gt 0 -and $iteration -eq 0) {
                    $allPatternData[$pattern.Name][$currentInputNeurons][$currentOutputNeurons] = @{
                        Value = $value
                        BlockConfig = $currentBlockConfig
                    }
                    Write-Host "  -> Stored data for $($pattern.Name): Input=$currentInputNeurons, Output=$currentOutputNeurons, Value=$value, BlockConfig=$currentBlockConfig"
                    $tmpvalue = $allPatternData[$pattern.Name][$currentInputNeurons][$currentOutputNeurons].Value
                    $tmpblkconfig = $allPatternData[$pattern.Name][$currentInputNeurons][$currentOutputNeurons].BlockConfig
                    Write-Host "  => Stored data for $($pattern.Name): Input=$currentInputNeurons, Output=$currentOutputNeurons, Value=$tmpvalue, BlockConfig=$tmpblkconfig"
                } else {
                    Write-Host "  -> Skipped: InputNeurons=$currentInputNeurons, OutputNeurons=$currentOutputNeurons, Iteration=$iteration"
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
        [string]$filename,
        [array]$InputNeurons,
        [array]$OutputNeurons,
        [string]$BaseFileName,
        [string]$CsvDirectory
    )
    
    $sortedInputNeurons = $InputNeurons | Sort-Object
    $sortedOutputNeurons = $OutputNeurons | Sort-Object
    
    Write-Host "DEBUG: Processing $PatternName"
    Write-Host "DEBUG: Input neurons to process: $($sortedInputNeurons -join ', ')"
    Write-Host "DEBUG: Output neurons to process: $($sortedOutputNeurons -join ', ')"
    Write-Host "DEBUG: Pattern data keys: $($PatternData.Keys -join ', ')"

    foreach ($currentInputNeurons in $InputNeurons) {
        foreach ($currentOutputNeurons in $OutputNeurons) {
            #$tmpvalue = $allPatternData[$patternName][$currentInputNeurons][$currentOutputNeurons].Value
            #$tmpblkconfig = $allPatternData[$patternName][$currentInputNeurons][$currentOutputNeurons].BlockConfig
            #Write-Host "  -- Stored data for $($patternName): Input=$currentInputNeurons, Output=$currentOutputNeurons, Value=$tmpvalue, BlockConfig=$tmpblkconfig"
            $tmpvalue = $PatternData[$currentInputNeurons][$currentOutputNeurons].Value
            $tmpblkconfig = $PatternData[$currentInputNeurons][$currentOutputNeurons].BlockConfig
            Write-Host "  -- Stored data for $($patternName): Input=$currentInputNeurons, Output=$currentOutputNeurons, Value=$tmpvalue, BlockConfig=$tmpblkconfig"

        }
    }
    
    #if ($sortedInputNeurons.Count -eq 0) {
        #Write-Host "No input neuron data found for pattern: $PatternName"
        #return $null
    #}
    
    #if ($sortedOutputNeurons.Count -eq 0) {
        #Write-Host "No output neuron data found for pattern: $PatternName"
        #return $null
    #}
    
    # Create CSV content
    $csv = New-Object System.Text.StringBuilder
    $csvFileName = Join-Path $CsvDirectory $BaseFileName
    
    # Build header row: "Input neurons/output neurons", then just the output neuron counts
    $headerRow = "Input neurons/output neurons"
    foreach ($currentOutputNeurons in $OutputNeurons) {
        $headerRow += ",$currentOutputNeurons"
    }
    [void]$csv.AppendLine($headerRow)
    
    # Build data rows: each row starts with input neuron count, then values for each output neuron count
    foreach ($currentInputNeurons in $InputNeurons) {
        $row = "$currentInputNeurons"
        Write-Host "DEBUG: Processing input neurons: $currentInputNeurons"
        
        foreach ($currentOutputNeurons in $OutputNeurons) {
            Write-Host "DEBUG: Checking for data at [$currentInputNeurons][$currentOutputNeurons]"
            $tmpvalue = $PatternData[$currentInputNeurons][$currentOutputNeurons].Value
            $tmpblkconfig = $PatternData[$currentInputNeurons][$currentOutputNeurons].BlockConfig
            Write-Host "  -- Stored data for $($patternName): Input=$currentInputNeurons, Output=$currentOutputNeurons, Value=$tmpvalue, BlockConfig=$tmpblkconfig"
            
            $dataPoint = $PatternData[$currentInputNeurons][$currentOutputNeurons]
            $value = $dataPoint.Value
            $row += ",$value"
            Write-Host "DEBUG: No data found for this combination"
        }
        [void]$csv.AppendLine($row)
        Write-Host "DEBUG: Row: $row"
    }
    
    # Save to CSV file
    $csvContent = $csv.ToString()
    Write-Host "DEBUG: Final CSV content for $PatternName :"
    Write-Host $csvContent
    
    $csvContent | Out-File -Encoding utf8 $csvFileName
    return $csvFileName
}

# Create output directory structure
$outputDir = Join-Path $csvDir $npuConfig
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    Write-Host "Created output directory: $outputDir"
}

# Sort the neuron arrays for consistent output
$allInputNeurons = $allInputNeurons | Sort-Object
$allOutputNeurons = $allOutputNeurons | Sort-Object
$allBlockConfigs = $allBlockConfigs | Sort-Object

# Process and save each pattern's data
$savedFiles = @()
foreach ($pattern in $patterns) {
    $patternData = $allPatternData[$pattern.Name]
    #foreach ($currentInputNeurons in $allInputNeurons) {
        #foreach ($currentOutputNeurons in $allOutputNeurons) {
            #$tmpvalue = $allPatternData[$pattern.Name][$currentInputNeurons][$currentOutputNeurons].Value
            #$tmpblkconfig = $allPatternData[$pattern.Name][$currentInputNeurons][$currentOutputNeurons].BlockConfig
            #Write-Host "  == Stored data for $($pattern.Name): Input=$currentInputNeurons, Output=$currentOutputNeurons, Value=$tmpvalue, BlockConfig=$tmpblkconfig"

        #}
    #}
    $tmpkeys = $patternData.Keys.Count
    Write-Host "$patternData.Keys.Count: $tmpkeys"
    #Write-Host "Before func call: Input neurons to process: $($sortedInputNeurons -join ', ')"
    #Write-Host "Before func call: Output neurons to process: $($sortedOutputNeurons -join ', ')"
    Write-Host "Before func call: Pattern data keys: $($PatternData.Keys -join ', ')"
    
    if ($patternData.Keys.Count -gt 0) {
        # Use just the suffix as filename in the dedicated directory
        $patternFileName = $pattern.filename + ".csv"
        $savedFile = Save-PatternToCSV -PatternData $patternData -PatternName $pattern.Name -Units $pattern.Units -filename $pattern.filename -InputNeurons $allInputNeurons -OutputNeurons $allOutputNeurons -BaseFileName $patternFileName -CsvDirectory $outputDir
        
        if ($savedFile) {
            $savedFiles += $savedFile
            Write-Host "$($pattern.Name) data saved to: $savedFile"
            
            # Display summary for this pattern
            Write-Host "  - Input neuron counts: $($allInputNeurons -join ', ')"
            Write-Host "  - Output neuron counts: $($allOutputNeurons -join ', ')"
            Write-Host "  - Block configurations (val3): $($allBlockConfigs -join ', ')"
            
            # Count total data points for this pattern
            $totalDataPoints = 0
            foreach ($inputNeurons in $patternData.Keys) {
                $totalDataPoints += $patternData[$inputNeurons].Count
            }
            Write-Host "  - Total data points: $totalDataPoints"
        }
    } else {
        Write-Host "No data captured for pattern: $($pattern.Name)"
    }
    Write-Host ""
}

# Overall summary
Write-Host "=== PROCESSING COMPLETE ==="
Write-Host "Total files created: $($savedFiles.Count)"
Write-Host "All input neuron sizes found: $($allInputNeurons -join ', ')"
Write-Host "All output neuron sizes found: $($allOutputNeurons -join ', ')"
Write-Host "All block configurations (val3) found: $($allBlockConfigs -join ', ')"
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