#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <sys/time.h>
#include <mpi.h>

//---------------------------------------MATRIX TRANSPOSITION CHECK---------------------------------------
bool checkTrans(int n, float **matrix, float **transpose){
   bool check=true;
   
   //check if the matrix has been transposed correctly
   for(int i=0; i<n; i++){
      for(int j=0; j<n; j++){
         if(matrix[i][j]!=transpose[j][i]){
            check = false;
         }
      }
   }
   
   return check;
}

//---------------------------------------PRINT MATRIX---------------------------------------
void print(int n, float **matrix){
   
   for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			printf("%f ", matrix[i][j]);
		}
		printf("\n");
   }
}

//---------------------------------------SYMMETRIC MATRIX MAKER---------------------------------------
void makeSymmetric(int n, float **matrix){
   for(int i=0; i<n; i++){
      for(int j=i+1; j<n; j++){
         matrix[i][j] = matrix[j][i];
      }
   }
}

//---------------------------------------SYMMETRY CHECK---------------------------------------
bool checkSym(int n, float **matrix){
   //suppose the matrix is symmetric
   bool check = true;
   
   double start, end;
   
   //start the wall-clock time
   start = MPI_Wtime();

   //symmetry check
   for(int i=0; i<n; i++){
      for(int j=i+1; j<n; j++){ //check only the upper part of the matrix
         //set the variable to false if the matrix isn't symmetric
         if(matrix[i][j]!=matrix[j][i]){
            check = false;
         }
      }
   }
   
   //stop the wall-clock time
   end = MPI_Wtime();
   
   double elapsed = end-start;
   
   //print results
   printf("\nSYMMETRY CHECK time = %12.4g sec\n", elapsed);
   
   return check;
}

//---------------------------------------MATRIX TRANSPOSITION---------------------------------------
double elapsed_serial;

float** matTranspose(int n, float **matrix){
   
   //allocate the transpose matrix
   float **transpose = (float **)malloc(n * sizeof(float *));
   if(transpose == NULL) {
      printf("Error.\n");
      return NULL;
   }
   for(int i=0; i<n; i++){
      transpose[i] = (float *)malloc(n * sizeof(float));
      if(transpose[i] == NULL){
         printf("Error.\n");
         return NULL;
      }
   }
   
   double start, end;
   start = MPI_Wtime();
   
   //transposition
   for(int i=0; i<n; i++){
      for(int j=0; j<n; j++){
         transpose[i][j] = matrix[j][i];
      }
   }

   end = MPI_Wtime();
      
   elapsed_serial = end-start;

   double bandwidth = (2.0*n*n*sizeof(float))/elapsed_serial/1e9;

   //print results
   printf("\nMATRIX TRANSPOSITION SERIAL time = %12.8f sec", elapsed_serial);
   printf("\nMATRIX TRANSPOSITION SERIAL bandwidth = %12.4f GB/sec\n", bandwidth);
   
   return transpose;
   
}

//---------------------------------------SYMMETRY CHECK MPI---------------------------------------
bool checkSymMPI(int n, float **matrix){
   
   int rank, size;
   bool local_check = true, global_check = true;
   
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   
   double start, end;
   
   //start the wall-clock time
   if(rank == 0) start = MPI_Wtime();
   
   //work distribution
   int local_rows = n/size;
   int start_row = rank*local_rows;
   int end_row = start_row+local_rows;

   //symmetry check
   for(int i=0; i<local_rows; i++){
      for(int j=i+1; j<n; j++){
         if(matrix[i][j]!=matrix[j][i]){
            local_check = false;
         }
      }
   }
   
   //reduction
   MPI_Reduce(&local_check, &global_check, 1, MPI_C_BOOL, MPI_LAND, 0, MPI_COMM_WORLD);
   
   if(rank == 0){
      //stop the wall-clock time
      end = MPI_Wtime();
      
      //time computation
      double elapsed = end-start;
      
      //print results
      printf("\nSYMMETRY CHECK MPI time = %12.4g sec\n", elapsed);
   }
   
   return global_check;
}

//---------------------------------------MATRIX TRANSPOSITION MPI---------------------------------------
float** matTransposeMPI(int n, float **matrix){

   double startT, endT, startF, endF;

   int rank, size;
   
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   
   if(rank==0) startF = MPI_Wtime();
   
   //work subdivision
   int local_rows = n/size;
   int start_row = rank*local_rows;
   int end_row = start_row+local_rows;
   
   //allocates the local buffer
   float *local_buffer = (float *)malloc(local_rows * n * sizeof(float));
   if(local_buffer==NULL){
      printf("Error.\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
   }
   
   //allocates the receiver buffer and the transpose matrix
   float **transpose = NULL;
   float *transpose_buffer = NULL;
   if(rank==0){
      transpose = (float **)malloc(n*sizeof(float *));
      for(int i=0; i<n; i++){
         transpose[i] = (float *)malloc(n*sizeof(float));
      }
      transpose_buffer = (float *)malloc(n*n*sizeof(float));
      if(transpose_buffer==NULL){
         printf("Error.\n");
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
   }
   
   if(rank==0) startT = MPI_Wtime();
   
   //local transposition
   for(int i=0; i<local_rows; i++){
      for(int j=0; j<n; j++){
         local_buffer[i*n+j] = matrix[j][start_row+i];
      }
   }
   
   if(rank==0) endT = MPI_Wtime();
   
   //uses MPI_Gather to collect the complete transpose
   MPI_Gather(local_buffer, local_rows*n, MPI_FLOAT, transpose_buffer, local_rows*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
   
   if(rank==0){
   
      for(int i=0; i<n; i++){
         transpose[i] = &transpose_buffer[i*n];
      }
   
      endF = MPI_Wtime();
      double elapsedT = endT-startT;
      double elapsedF = endF-startF;

      double bandwidth = (2.0 * n * n * sizeof(float)) / elapsedT / 1e9;
      double speedup = elapsed_serial/elapsedT;
      double efficiency = speedup/size;
      

      printf("\nMATRIX TRANSPOSITION MPI time (ONLY TRANSPOSITION) = %12.8f sec", elapsedT);
      printf("\nMATRIX TRANSPOSITION MPI time (ALL THE FUNCTION) = %12.8f sec", elapsedF);
      printf("\nMATRIX TRANSPOSITION MPI bandwidth = %5.4f GB/sec", bandwidth);
      printf("\nMATRIX TRANSPOSITION MPI speedup = %5.2f", speedup);
      printf("\nMATRIX TRANSPOSITION MPI efficiency = %5.2f%%", efficiency*100);
      printf("\n");
      
      //open CSV file for writing results
      FILE *file = fopen("results.csv", "a");
      if (file == NULL) {
         perror("Failed to open file");
         return NULL;
      }
      
      //write results to CSV
      fprintf(file, "%11d;%11d;%12.8f;%12.8f;%5.4f;%5.2f;%5.2f\n", 
                     size, n, elapsedT, elapsedF, bandwidth, speedup, efficiency);
      //close CSV file
      fclose(file);
      
      free(local_buffer);

   }
   
   return transpose;
}

//---------------------------------------------MAIN---------------------------------------------
int main(int argc, char *argv[]) {
   
   int n = atoi(argv[1]); //converts a string passed as a command line argument into an integer 
   //check if n is a power of 2
   if(!((n>0)&&((n&(n-1))==0))){ 
      return 1;
   }
   
   bool check;
   
   MPI_Init(&argc, &argv);
   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   
   if((n%size)!=0){
      if(rank==0){
         printf("The matrix size must be divisible by the number of processes\n");
      }
      MPI_Finalize();
      return 1;
   }
   
   //allocate matrix
   float **m = (float **)malloc(n * sizeof(float *));
   if(m == NULL) {
      printf("Error.\n");
      return 1;
   }
   
   for(int i=0; i<n; i++){
      m[i] = (float *)malloc(n * sizeof(float));
      if(m[i] == NULL){
         printf("Error.\n");
         return 1;
      }
   }
   
   //allocate transpose matrix
   float **t = (float **)malloc(n * sizeof(float *));
   if(t == NULL) {
      printf("Error.\n");
      return 1;
   }
   
   for(int i=0; i<n; i++){
      t[i] = (float *)malloc(n * sizeof(float));
      if(t[i] == NULL){
         printf("Error.\n");
         return 1;
      }
   }
 
   srand(time(NULL));
 
   //INITIALIZATION
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			m[i][j] = ((float)rand()/RAND_MAX)*10; //random float number from 0 to 10
		}
	}
 
   //makeSymmetric(n, m); //makes the matrix symmetric
 
   if(rank==0){
   
      //PRINT THE MATRIX
      //print(n, m);
      //printf("\n");
      
      printf("\n-----------------------------SYMMETRY CHECK-----------------------------\n");
      
      //CHECKSYM SERIAL
      printf("\nSYMMETRY CHECK SERIAL\n");
      check = checkSym(n, m);
      if(check) printf("\nSYMMETRIC\n"); else printf("\nNOT symmetric\n");
      
      //CHECKSYM MPI
      printf("\nSYMMETRY CHECK MPI\n");
   }
   
   check = checkSymMPI(n, m);
   
   if(rank==0){
      if(check) printf("\nSYMMETRIC\n"); else printf("\nNOT symmetric\n");
   
      printf("\n--------------------------MATRIX TRANSPOSITION--------------------------\n");
         
      //MATTRANSPOSE SERIAL
      t = matTranspose(n, m);
      if(checkTrans(n, m, t)) printf("\nCORRECT\n"); else printf("\nINCORRECT\n"); //check if the transposition is correct
      //print(n, t);
   }
   
   //MATTRANSPOSE MPI
   t = matTransposeMPI(n, m);
   
   if(rank==0){
      if(t!=NULL && checkTrans(n, m, t)) printf("\nCORRECT\n"); else printf("\nINCORRECT\n"); //check if the transposition is correct
      //print(n, t);  
   }
   
   MPI_Finalize();
   
   //free the matrices
   for(int i=0; i<n; i++){
      free(m[i]);
   }
   free(m);
   free(t);
  
	return 0;
}