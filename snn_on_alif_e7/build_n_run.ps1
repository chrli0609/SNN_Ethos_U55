# Enable PowerShell to stop on the first error
$ErrorActionPreference = "Stop"

# Run the cmsis-csolution.build command first
Write-Host "Running cmsis-csolution.build..."
try {
    cbuild C:\Users\chris\skolsaker\ak5\mex\project\snn_ethosu\mlek.csolution.yml --context-set --packs
}
catch { 
    Write-Host "Build Failed, exiting..."
    exit
}




$outputElfFilename='C:\Users\chris\skolsaker\ak5\mex\project\snn_ethosu\out\simple_code_test\Alif-E7-M55-HP\debug\simple_code_test.elf'; $outputBinFilename=${outputElfFilename}.Substring(0, $outputElfFilename.Length -3) + 'bin'; cp "$outputBinFilename" '/Users/chris/tools/Alif Security Toolkit/APFW0001-app-release-exec-windows-SE_FW_1.103.00_DEV/app-release-exec/build/images/alif-img.bin'; cp './.alif/M55_HP_cfg.json' '/Users/chris/tools/Alif Security Toolkit/APFW0001-app-release-exec-windows-SE_FW_1.103.00_DEV/app-release-exec/alif-img.json';



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
    echo "COM8" | ./app-write-mram -p -d

    rm ./build/images/alif-img.bin
    rm ./alif-img.json
}
catch { Set-Location -Path $originalPath } # return to original directory if error so we dont get stuck in $ToolkitPath
finally { Set-Location -Path $originalPath }



# Return to the original directory
Write-Host "Returning to the original directory..."
Set-Location -Path $originalPath

Write-Host "Script execution completed!"



#$outputElfFilename='c:\Users\chris\skolsaker\ak5\mex\project\alif\e7_simple_program\out\simple_code_test\Alif-E7-M55-HP\debug\simple_code_test.elf'; $outputBinFilename=${outputElfFilename}.Substring(0, $outputElfFilename.Length -3) + 'bin'; cp "$outputBinFilename" '/Users/chris/tools/Alif Security Toolkit/APFW0001-app-release-exec-windows-SE_FW_1.103.00_DEV/app-release-exec/build/images/alif-img.bin'; cp './.alif/M55_HP_cfg.json' '/Users/chris/tools/Alif Security Toolkit/APFW0001-app-release-exec-windows-SE_FW_1.103.00_DEV/app-release-exec/alif-img.json'; 
 
#c:\Users\chris\tools\'Alif Security Toolkit'\APFW0001-app-release-exec-windows-SE_FW_1.103.00_DEV\app-release-exec\app-gen-toc.exe -f c:\Users\chris\tools\'Alif Security Toolkit'\APFW0001-app-release-exec-windows-SE_FW_1.103.00_DEV\app-release-exec\alif-img.json
#./app-gen-toc -f alif-img.json; ./app-write-mram -p -d; rm ./build/images/alif-img.bin; rm ./alif-img.json; 


#$answer = Read-Host 'Read from COM7?'

#if($answer -eq 'y'){
.\scripts\read_com7.ps1
#}
