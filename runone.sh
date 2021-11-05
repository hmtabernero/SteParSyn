echo $1 > infile
echo $2 >> infile
echo $3 >> infile
python3 SteParSyn.py  < infile  #> $1.log
