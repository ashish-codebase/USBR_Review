powershell -Command "& {Get-ChildItem -Path '.git' -Recurse -File | Remove-Item -Force}"
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/ashish-codebase/USBR_Review.git
git branch -M main
git push --force -u origin main
git-auto-sync sync
