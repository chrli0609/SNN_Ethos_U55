
param (
    [string]$model,
    [int]$modifyPercent = 75,
    [string]$PythonScriptDir = "..\ethosu_compiler\gen_cms\",
    [string]$PythonScriptFilename = "main_repeat_weights.py",
    [string]$SecondaryScriptFilename = "build_n_run.ps1"
)



$originalPath = Get-Location










for ($i = 5; $i -lt 101; $i++) {

Set-Location -Path $PythonScriptDir
try {

    #cd $PythonScriptDir
    #$pythonOutput = & python $PythonScriptFilename $outputNeurons 2>&1
    $pythonOutput = python $PythonScriptFilename --model $model --modify-percent $modifyPercent --num_hidden $i
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




# Set the right model in cproject file
try {
    $editprojOutput = python3 .\scripts\edit_cproject_file.py  --model $model
}
catch {
    Write-Error "Failed when running edit_cproject_file.py"
    exit 1
}

if ($editprojOutput -match "Traceback") {
    Write-Error "Error found in edit cproject file python script"
    exit 1
}


# Run the secondary PowerShell script if provided
$scriptOutput = .\build_n_run.ps1




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


Write-Output "Wrote to MRAM Successfully, will wait 60 seconds for the program to run before starting next iteration"


Start-Sleep -Seconds 60

}