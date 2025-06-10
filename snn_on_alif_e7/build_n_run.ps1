# Enable PowerShell to stop on the first error
$ErrorActionPreference = "Stop"

# Run the cmsis-csolution.build command first
Write-Host "Running cmsis-csolution.build..."
try {
    $build_output = cbuild C:\Users\chris\skolsaker\ak5\mex\project\SNN_Ethos_U55\snn_on_alif_e7\mlek.csolution.yml --context-set --packs
}
catch { 
    Write-Host "Build Failed, exiting..."
    exit
}
echo $build_output

if ($build_output -match "Build summary: 1 succeeded, 0 failed") {
    Write-Host "Build succeeded!"
} elseif ($build_output -match "Build summary: 0 succeeded, 1 failed") {
    Write-Host "Build failed!"
    exit
} else {
    Write-Host "No relevant build summary found."
    exit
}



$outputElfFilename='C:\Users\chris\skolsaker\ak5\mex\project\SNN_Ethos_U55\snn_on_alif_e7\out\my_snn_bare_metal\Alif-E7-M55-HP\debug\my_snn_bare_metal.elf'; $outputBinFilename=${outputElfFilename}.Substring(0, $outputElfFilename.Length -3) + 'bin'; cp "$outputBinFilename" '/Users/chris/tools/Alif Security Toolkit/APFW0001-app-release-exec-windows-SE_FW_1.103.00_DEV/app-release-exec/build/images/alif-img.bin'; cp './.alif/M55_HP_cfg.json' '/Users/chris/tools/Alif Security Toolkit/APFW0001-app-release-exec-windows-SE_FW_1.103.00_DEV/app-release-exec/alif-img.json';



# Save the initial directory
$originalPath = Get-Location

# Define the working directory for the Alif Security Toolkit
$ToolkitPath = "C:\Users\chris\tools\Alif Security Toolkit\APFW0001-app-release-exec-windows-SE_FW_1.103.00_DEV\app-release-exec"


# Change to the toolkit directory
Write-Host "Navigating to Alif Security Toolkit directory..."
Set-Location -Path $ToolkitPath


try {


    # Run ./app-gen-toc -f alif-img.json
    Write-Host "Generating TOC..."

    # Run ./app-write-mram -p -d and send "COM8" as input manually
    Write-Host "Writing to MRAM..."

    # Start the app-write-mram process and ensure it's running
    ./app-gen-toc -f alif-img.json
    #echo "COM8" | ./app-write-mram -p -d
    #echo "COM12" | ./app-write-mram -p -d
    echo "COM32" | ./app-write-mram -p -d

    rm ./build/images/alif-img.bin
    rm ./alif-img.json
}
catch { Set-Location -Path $originalPath } # return to original directory if error so we dont get stuck in $ToolkitPath
finally { Set-Location -Path $originalPath }



# Return to the original directory
Write-Host "Returning to the original directory..."
Set-Location -Path $originalPath

Write-Host "Script execution completed!"

