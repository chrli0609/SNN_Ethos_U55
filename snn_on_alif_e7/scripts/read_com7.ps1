$port = New-Object System.IO.Ports.SerialPort COM7,115200,None,8,one
#$port = New-Object System.IO.Ports.SerialPort COM8,55000,None,8,one
$port.Encoding = [System.Text.Encoding]::UTF8

$port.NewLine     = "`n"             # Expect each packet to end in LF
$port.ReadTimeout = 500              # Milliseconds before ReadLine() throws

$port.Open()

Write-Host "Press 'q' and Enter to stop reading..."
while ($port.IsOpen) {
    #if ($port.BytesToRead -gt 0) {
        ##$data = $port.ReadExisting()  # Read all available data, no newline needed
        ##Write-Output $data
    #}

    try{

        $line = $port.ReadLine()
        Write-Output $line
        # If we ever see exactly "End of main() reached", quit:
        if ($line -eq "End of main() reached\n") {

            Write-Output "entered if statement for breaking"
            break
        }
    }

    catch [System.TimeoutException] {
        # No full line arrived in 500 ms—fall through to check for 'q'
    }

    # Check if user pressed 'q' to exit
    if ([console]::KeyAvailable) {
        $key = [console]::ReadKey($true).Key
        if ($key -eq "Q") {
            break
        }
    }

    Start-Sleep -Milliseconds 1
}

$port.Close()
Write-Host "Port closed."

