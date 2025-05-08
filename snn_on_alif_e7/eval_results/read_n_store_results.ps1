# Serial Port Reader with CSV Export
# This script reads data from COM7, parses neuron count and tick data, and exports to CSV

# Configure and open the serial port
$port = New-Object System.IO.Ports.SerialPort COM7,115200,None,8,one
$port.Open()

# Data storage
$allData = @{}
$currentNeurons = 0
$rawOutput = ""

Write-Host "Reading from COM7. Press 'q' and Enter to stop reading..."

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
        $allData[$currentNeurons] = @{}
    }
    # Check for tick data line
    elseif ($line -match "Ticks elapsed for layer once in it: (\d+) = (\d+)") {
        $iteration = [int]$matches[1]
        $tickValue = [int]$matches[2]
        
        if ($currentNeurons -gt 0) {
            $allData[$currentNeurons][$iteration] = $tickValue
        }
    }
}

# Find all unique neuron counts and iterations
$neuronCounts = $allData.Keys | Sort-Object
$maxIteration = 0
foreach ($neurons in $neuronCounts) {
    $iterations = $allData[$neurons].Keys | Sort-Object
    $maxIteration = [Math]::Max($maxIteration, ($iterations | Measure-Object -Maximum).Maximum)
}

# Create CSV content
$csv = New-Object System.Text.StringBuilder
$csvFileName = "neuron_ticks_data_$(Get-Date -Format 'yyyyMMdd_HHmmss').csv"

# Build header row with all neuron counts
$headerRow = "Iteration"
foreach ($neurons in $neuronCounts) {
    $headerRow += ",neurons_$neurons"
}
[void]$csv.AppendLine($headerRow)

# Build data rows
for ($i = 0; $i -le $maxIteration; $i++) {
    $row = "$i"
    foreach ($neurons in $neuronCounts) {
        if ($allData[$neurons].ContainsKey($i)) {
            $row += ",$($allData[$neurons][$i])"
        } else {
            $row += ","  # Empty value if this iteration doesn't exist for this neuron count
        }
    }
    [void]$csv.AppendLine($row)
}

# Save to CSV file
$csv.ToString() | Out-File -Encoding utf8 $csvFileName

Write-Host "Data has been processed and saved to $csvFileName"

# Display summary of what was processed
Write-Host "Summary:"
Write-Host "- Number of different neuron configurations: $($neuronCounts.Count)"
Write-Host "- Neuron counts found: $($neuronCounts -join ', ')"
Write-Host "- Maximum iteration number: $maxIteration"



& python plot_csv.py $csvFileName