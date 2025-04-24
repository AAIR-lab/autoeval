/*
(Classical) Random k-CNF Generator.
Authors:
  - Jesús Giráldez-Cru (UGR)

Contact: jgiraldez@ugr.es

*/

#include <getopt.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <stdio.h>

using namespace std;

int k = 3;		// k-CNF
int seed = 0;	
int n = 10000;	// number of variables
int m = 42500;	// number of clauses

char* output = NULL;

void printUsage(char* app){
	cerr << "  Usage: " << app << " [options]" << endl;
	cerr << "    -n <int>   :  number of variables (10000)" << endl;
	cerr << "    -m <int>   :  number of clauses (42500)" << endl;
	cerr << "    -k <int>   :  number of literals by clause: k-CNF (3)" << endl;
	cerr << "    -s <int>   :  seed (0)" << endl;
	cerr << "    -o <string>:  output file (stdout)" << endl;
	exit(-1);
}

void parseArgs(int argc, char **argv){
	int opt;
	while((opt=getopt(argc, argv, "n:m:s:k:?ho:")) != -1){
		switch(opt){
			case 'n':
				n = atoi(optarg);
				break;
			case 'm':
				m = atoi(optarg);
				break;
			case 's':
				seed = atoi(optarg);
				break;
			case 'k':
				k = atoi(optarg);
				if(k < 1){
					cerr << "WARNING: k must be greater than 0" << endl;
					cerr << "  k changed to 3 (default value)" << endl;
					k = 3;
				}
				break;
			case 'o':
				output = optarg;
				break;
			case 'h':
			case '?':
				printUsage(argv[0]);
				break;
			default:
				cerr << "ERROR: Incorrect argument: " << optarg << endl;
				printUsage(argv[0]);
		}
	}
}



int main(int argc, char **argv){
	
	FILE *fout;
	
	// Parse arguments
	parseArgs(argc, argv);
	
	if(output!=NULL){
		fout = fopen(output, "w");
	}
	
	
	// Print header
	if(output!=NULL){
		fprintf(fout, "c Random k-CNF Generator. J. Giraldez-Cru\n");
		fprintf(fout, "c   value n = %d\n", n);
		fprintf(fout, "c   value m = %d\n", m);
		fprintf(fout, "c   value k = %d\n", k);
		fprintf(fout, "c   value seed = %d\n", seed);
		fprintf(fout, "p cnf %d %d\n", n, m);
	}else{
		cout << "c Random k-CNF Generator. J. Giraldez-Cru" << endl;
		cout << "c   value n = " << n << endl;
		cout << "c   value m = " << m << endl;
		cout << "c   value k = " << k << endl;
		cout << "c   value seed = " << seed << endl;
		cout << "p cnf " << n << " " << m << endl;		
	}

	
	int rn;			// random number
	double rd;		// random double between 0 and 1
	srand (seed);
	
	// Iterate for each clause
	for(int i=0; i<m; i++){
		
		
		// Compute the clause
		vector<int> clause(k);
		
		for(int j=0; j<k; j++){
			// Random variable avoiding tautologies with previous literals
			int var;
			bool tautology = false;
			
			do{
				tautology = false;
				
				rn = rand();
				var = rn % n + 1;
				
				for(int l=0; l<j && !tautology; l++){
					if(abs(clause[l]) == var){
						tautology = true;
					}
				}
				
			}while(tautology);
		
			// Random sign of the variable
			if(rn > (RAND_MAX/2))
				var = -var;
		
			// Save it in the clause
			clause[j] = var;
		}
		
		// Print the clause
		for(int j=0; j<k; j++){
			if(output!=NULL){
				fprintf(fout, "%d ", clause[j]);
			}else{
				cout << clause[j] << " ";
			}
		}
		
		if(output!=NULL){
			fprintf(fout, "0\n");
		}else{
			cout << "0" << endl;
		}
		
	}
	
	if(output!=NULL){
		fclose(fout);
	}
}