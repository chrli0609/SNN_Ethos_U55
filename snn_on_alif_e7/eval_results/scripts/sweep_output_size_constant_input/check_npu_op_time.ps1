param (
    [string]$PythonScriptDir,# = "../ethosu_compiler/gen_cms/mlp_int8/",
    [int]$inputNeurons,
    [string]$PythonScriptFilename = "main.py",
    [int]$OutputNeuronsStartRange = 16,
    [int]$OutputNeuronsEndRange = 512+16,
    #[string]$SecondaryScriptDir = "./",
    [string]$SecondaryScriptFilename = "build_n_run.ps1"
)



#Store current directory
$originalPath = Get-Location

# Run the Python script for each value in the range
#foreach ($outputNeurons in $outputNeuronsStartRange..$outputNeuronsEndRange) {
for ($outputNeurons = $outputNeuronsStartRange; $outputNeurons -le $outputNeuronsEndRange; $outputNeurons += 16) {

    Write-Host "Processing with outputNeurons = $outputNeurons" -ForegroundColor Green
    

    # Run the Python script with the current value as argument
    Set-Location -Path $PythonScriptDir
    try {

    	#cd $PythonScriptDir
    	#$pythonOutput = & python $PythonScriptFilename $outputNeurons 2>&1
    	$pythonOutput = python $PythonScriptFilename $inputNeurons $outputNeurons
        Write-Output "This is python output!"
        Write-Output "$pythonOutput"
    	#$pythonOutput = & python3 $PythonScriptFilename
    }
    catch { 
        Set-Location -Path $originalPath
        echo "Python Script Failed!"
        exit 1
    }
    finally {
        Set-Location -Path $originalPath
        echo "Python Script Ran Successfully!"
    }
    

    if ($pythonOutput -match "Error") {
        Write-Error "Error found in python script"
        exit 1
    }
    
    if ($pythonOutput -match "Traceback") {
        Write-Error "Error found in python script"
        exit 1
    }



    # Run the secondary PowerShell script if provided
    #Set-Location -Path $SecondaryScriptDir
    #try {
    	#$scriptOutput = & $SecondaryScript 2>&1
    #$scriptOutput = &"$SecondaryScriptFilename"
    $scriptOutput = .\build_n_run.ps1
    #}
    #catch { set-Location -Path $originalPath }
    #finally { Set-Location -Path $originalPath }

    #echo "$scriptOutput"
        
    # Check for build failure in secondary script output
    #Write-Output "about to enter Check-BuildFailure"
    #if (Check_ScriptFailure -Output $scriptOutput) {
        ##[System.Environment]::Exit(1)
        #Write-Output "Check-BuildFailure Evaluated to true"
        #
    #}
    
    #Write-Output "This is output!!!"
    #Write-Output "$scriptOutput"




    if ($scriptOutput -match "Build summary: 0 succeeded, 1 failed") {
        Write-Error "Build failure detected! Stopping execution."
        Write-Output "----------------------------------------------------------------------------------------------------"
        exit 1
    }

    if ($scriptOutput -match "ERROR") {
        Write-Error "Failed to write to COM port."
        Write-Output "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        exit 1
    }


}

#Write-Host "All processing complete. Results saved to $csvPath" -ForegroundColor Green
Write-Host "All processing complete." -ForegroundColor Green