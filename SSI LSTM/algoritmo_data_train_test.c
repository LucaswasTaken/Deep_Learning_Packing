#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;
int main(int argc, char** argv)
{
    //Declaracao das Variaveis de Auxilio
    int altura = 8;
    int largura = 8;
    int pcord;
    int r_particle = 1;
    //Geracao aleatoria das particulas
    int n_samples = 12000;
    int n_particles = 15;
    int particles[n_samples][altura*largura];
    int aux[n_samples][n_particles];
    int verifica = 0;
    int count;
    int count2 = 0;
    int verifica2 = 0;
    for (int i = 0; i < n_samples; ++i)
    {
        for (int j = 0; j < altura*largura; ++j)
        {
            particles[i][j] = 0;
        }
    }
    for (int i = 0; i < n_samples ; ++i) 
    {
        cout<<count2<<endl;
        srand(count2);
        count = 0;
        for (int j = 0; j < n_particles; ++j)
        {      	
            pcord = (rand()%(altura*largura));
            aux[i][j] = pcord;
            if (j==0)
            {
                particles[i][pcord] = 1;
            }
            else
            {
            verifica = 0;
            for (int k = 0; k < j; ++k)
                {
                    if(abs(aux[i][k]-pcord)<=(altura+2*r_particle))
                    {
                        if((abs((aux[i][k]%altura)-(pcord%altura))<2*r_particle))
                        {
                            j = j-1;
                            verifica = 1;
                            count = count+1;
                            if(count==100)
                            {
                            	count = 0;
                            	verifica2 =1;
                            }
                            break;
                        }
                    }
                }
                if (verifica ==0)
                {
                    //cout<<j<<endl;
                    count = 0;
                    particles[i][pcord] = 1;
                	count2 = count2+1;
                }
                if (verifica2 == 1)
                {
                	verifica2=0;
                	i = i-1;
                	count2 = count2+1;
                	break;
                }
            }
        }
    }
    ofstream myfile;
    myfile.open ("traindata.txt");
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_particles; ++j)
        {
            myfile <<aux[i][j]<<" ";
        }
        myfile <<endl;     
    }
    myfile.close();
    return 0;
}
  