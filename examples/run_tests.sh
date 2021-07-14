#!/bin/bash

for d in 128 512
do
	for n in 1000000
	do
		for M in 24 48 64
		do
			for ef in 500 1000
			do
				k=${M}
				res=$(./test_hnsw d ${d} n ${n} k ${k} M ${M} ef_c ${ef} ef ${ef})
				if (( $(echo "$res > 0.9" |bc -l) )); then
					break
				fi
			done
		done
	done
done


