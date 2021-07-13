#!/bin/bash

for d in 4 32 128 512
do
	for n in 1000 10000 100000 1000000
	do
		for M in 6 12 24 48 64
		do
			for ef in 10 100 200 500
			do
				k=${M}
				if [[ ${M} -gt ${ef} ]]; then
				 	k=${ef}
				fi
				res=$(./test_hnsw d ${d} n ${n} k ${k} M ${M} ef_c ${ef} ef ${ef})
				if (( $(echo "$res > 0.9" |bc -l) )); then
					break
				fi
			done
		done
	done
done
