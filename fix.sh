find . -type f -iname 'conop9.cfg' | while read -r file; do
  dir=$(dirname "$file")
  base=$(basename "$file")
  # 检查同目录下是否有大写版本
  if [ -f "$dir/CONOP9.CFG" ] && [ "$file" != "$dir/CONOP9.CFG" ]; then
    # 如果当前是小写，且有大写，重命名大写为 .BAK
    mv "$dir/CONOP9.CFG" "$dir/CONOP9.CFG.BAK"
    echo "Renamed $dir/CONOP9.CFG to $dir/CONOP9.CFG.BAK"
  fi
done