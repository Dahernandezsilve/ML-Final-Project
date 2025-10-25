$folders = @(
  "business_understanding",
  "data_understanding",
  "data_preparation",
  "modeling",
  "evaluation",
  "deployment"
)

foreach ($f in $folders) {
  Write-Host "== Instalando $f ==" -ForegroundColor Cyan
  Set-Location $f
  pip install -e .
  Set-Location ..
}
