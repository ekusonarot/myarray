for i in 1 2
do
echo test${i}
cat ./input/in${i}.txt | python3 main.py > ./result/result${i}.txt
diff ./output/out${i}.txt ./result/result${i}.txt -u
done