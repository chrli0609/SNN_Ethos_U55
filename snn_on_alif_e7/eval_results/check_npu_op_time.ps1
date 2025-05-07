param (
    [string]$PythonScriptDir = "../ethosu_compiler/gen_cms/mlp_int8/",
    [string]$PythonScriptFilename = "main.py",
    [int]$StartRange = 16,
    [int]$EndRange = 1024,
    #[string]$SecondaryScriptDir = "./",
    [string]$SecondaryScriptFilename = "build_n_run.ps1"
)

# Function to check for "Build failed!" pattern
function Check-BuildFailure {
    param (
        [string]$Output
    )
    
    if ($Output -match "Build failed!") {
        Write-Error "Build failure detected! Stopping execution."
        return $true
    }
    return $false
}

# Setup global variables for background job to read serial port
$script:stopReading = $false
$script:serialData = ""
$script:currentNeurons = 0
$script:allResults = @{}
$script:buildFailed = $false








# Create CSV file to store results
$csvPath = "neuron_performance_results.csv"
$maxIterations = 100  # Assuming we'll collect up to 100 iterations

# Create CSV header (Neuron count as columns)
$csvHeader = "Iteration,"
for ($i = $StartRange; $i -le $EndRange; $i++) {
    $csvHeader += "$i,"
}
$csvHeader = $csvHeader.TrimEnd(',')
Set-Content -Path $csvPath -Value $csvHeader

# Create a dictionary to store all results
$allResults = @{}

#Store current directory
$originalPath = Get-Location

# Run the Python script for each value in the range
foreach ($neurons in $StartRange..$EndRange) {
    Write-Host "Processing with neurons = $neurons" -ForegroundColor Green
    

    # Run the Python script with the current value as argument
    Set-Location -Path $PythonScriptDir
    try {

    	#cd $PythonScriptDir
    	#$pythonOutput = & python $PythonScript $neurons 2>&1
    	$pythonOutput = & python3 $PythonScriptFilename
    }
    catch { Set-Location -Path $originalPath }
    finally {Set-Location -Path $originalPath }
    

    # Check for build failure
    if (Check-BuildFailure -Output ($pythonOutput -join "`n")) {
        break
    }
    




    # Run the secondary PowerShell script if provided
    #Set-Location -Path $SecondaryScriptDir
    #try {
	echo "just beofre calling secondary script"
    	#$scriptOutput = & $SecondaryScript 2>&1
    #$scriptOutput = &"$SecondaryScriptFilename"
    $scriptOutput = .\build_n_run.ps1
	echo "we just ran secondary script!!!"
    #}
    #catch { set-Location -Path $originalPath }
    #finally { Set-Location -Path $originalPath }


        
    # Check for build failure in secondary script output
    if (Check-BuildFailure -Output ($scriptOutput -join "`n")) {
    	break
    }
    



<#
    # Read from COM port and collect results
    $results = Read-SerialPort -CurrentNeurons $neurons
    
    # Store results
    if ($results.TicksData.Count -gt 0) {
        $allResults[$neurons] = $results.TicksData
    }
    else {
        Write-Warning "No tick data collected for neurons = $neurons"
    }
    
    # Update CSV after each run
    for ($iteration = 0; $iteration -le $maxIterations; $iteration++) {
        $csvLine = "$iteration,"
        
        for ($n = $StartRange; $n -le $EndRange; $n++) {
            if ($allResults.ContainsKey($n) -and $allResults[$n].ContainsKey($iteration)) {
                $csvLine += "$($allResults[$n][$iteration]),"
            }
            else {
                $csvLine += ","
            }
        }
        
        $csvLine = $csvLine.TrimEnd(',')
        
        # Update specific line in CSV
        if ($iteration -eq 0) {
            # Check if file already has content beyond header
            $currentLines = Get-Content -Path $csvPath
            if ($currentLines.Count -le 1) {
                Add-Content -Path $csvPath -Value $csvLine
            }
            else {
                $currentLines[1] = $csvLine
                Set-Content -Path $csvPath -Value $currentLines
            }
        }
        elseif ($iteration -lt $currentLines.Count - 1) {
            $currentLines[$iteration + 1] = $csvLine
            Set-Content -Path $csvPath -Value $currentLines
        }
        else {
            Add-Content -Path $csvPath -Value $csvLine
        }
    }
#>
}

#Write-Host "All processing complete. Results saved to $csvPath" -ForegroundColor Green
Write-Host "All processing complete." -ForegroundColor Green