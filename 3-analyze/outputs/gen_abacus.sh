#!/bin/bash


#python generate_plots.py oxides-verification-PBE-v1 'FLEUR@LAPW+LO'
python generate_plots.py abacus_c19mk2 unaries-verification-PBE-v1 'FLEUR@LAPW+LO'
python generate_plots.py abacus_c19mk2 unaries-verification-PBE-v1 'CASTEP@PW|C19MK2'

for i in epsilon V0_rel_diff B1_rel_diff B0_rel_diff; do

#python generate_histos.py oxides-verification-PBE-v1 $i 'FLEUR@LAPW+LO'
python generate_histos.py abacus_c19mk2 unaries-verification-PBE-v1 $i 'FLEUR@LAPW+LO'
python generate_histos.py abacus_c19mk2 unaries-verification-PBE-v1 $i 'CASTEP@PW|C19MK2'
done
