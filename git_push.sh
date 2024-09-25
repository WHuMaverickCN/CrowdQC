# just push branch -main
git add .
git commit -m "batch compare recons to gt"
git push -u origin_caq main

# addtional push dev and merge main to dev
git checkout dev 
git merge main
git branch -vv
git push -u origin_caq dev
#   temp

#   git rm temp_folder -r
#   更新gitclone