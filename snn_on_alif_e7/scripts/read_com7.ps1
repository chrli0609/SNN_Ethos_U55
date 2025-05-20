$port = New-Object System.IO.Ports.SerialPort COM7,115200,None,8,one
#$port = New-Object System.IO.Ports.SerialPort COM8,55000,None,8,one
$port.Encoding = [System.Text.Encoding]::UTF8
$port.Open()

Write-Host "Press 'q' and Enter to stop reading..."
while ($port.IsOpen) {
    if ($port.BytesToRead -gt 0) {
        $data = $port.ReadExisting()  # Read all available data, no newline needed
        Write-Output $data
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
Write-Host "Port closed."

