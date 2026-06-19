$root = "E:\spacecraft_ai_project"
$src = "$root\print_library"

$files = Get-ChildItem $src -Recurse -Filter *.md

foreach ($file in $files) {
    Write-Host "Exporting $($file.FullName)"

    code --reuse-window $file.FullName
    Start-Sleep -Seconds 2

    code --reuse-window --command "markdown-pdf.convertPdf"
    Start-Sleep -Seconds 4
}

Write-Host "Done."