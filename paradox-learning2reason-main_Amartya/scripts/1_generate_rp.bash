FOLDER=DATA/RP

mkdir -p $FOLDER
for number in 0 1 2 3 4 5 6 7 8 9  # sample with 10 processes
do
   python sample/sample.py --vocab_file sample/vocab.txt --output_file $FOLDER/prop_examples_$number.txt --min_pred_num 5 --max_pred_num 30 --algo RP --example_num 10 --balance_by_depth --max_depth 6 & 
done
wait
python dataset.py --file_name $FOLDER/prop_examples_INDEX.txt --file_range 10 --max_depth_during_train 6 --final_file_name $FOLDER/prop_examples.balanced_by_backward.max_6.json --control_num 10 --depth depth