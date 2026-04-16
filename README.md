### To generate data:
- _Note: Basic compilation instructions are at top of each file_
- Compile and run gen.cpp with your preferred number of events, and number of particles per event
- Compile and run calc.cpp (using `events.txt` generated in previous step)
- Proceed with next steps using `events_with_4vec.txt` as input for further processing
	- Columns here are: pT eta phi mass id px py pz E
- _Also note: If we decide to time / parallelize this stage, we should convert to C_
