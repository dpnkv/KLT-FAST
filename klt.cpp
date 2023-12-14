#include <iostream>
#include <vector>
#include <cmath>


void print_vec(std::vector<std::vector<int>> &vec)
{
    for (std::vector<int> vect1D : vec)
    {
        for (int x : vect1D)
        {
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
}

// void matrix_inverse(std::vector<std::vector<int>> &G);
// void matrix_multiply(std::vector<std::vector<int>> &G, std::vector<std::vector<int>> &bk);

int main()
{
    const int nx = 40, ny = 20; // original image size
    const int Lm = 2; // deepest layer of pyramid
    const int omega_x = 2, omega_y = 2; // 
    const int K = 10; // number of iterations for iterative L-K

    // vector for storing image dimensions for different layers
    // pair of [nx ny] for L-th layer: [2*L 2*L+1]
    std::vector<int> dimensions((Lm+1) * 2);
    dimensions[0] = nx;
    dimensions[1] = ny;
    for (int x : dimensions)
        std::cout << x << " ";
    
    // vector for greyscale values of original image
    // 3dvec[L] = std::vector<std::vector<int>>(dimensions[2*L], std::vector<int>dimensions[2*L+1])
    // for loop for copying calculated I on step L into 3dvec[L]
    std::vector<std::vector<std::vector<int>>> I(Lm);
    std::vector<std::vector<std::vector<int>>> J(Lm);
 
    // I[0] = std::vector<std::vector<int>>(dimensions[0], std::vector<int>(dimensions[1]));
    
    // filling I and J with random int numbers from [0, 255]
    for (int i = 0; i < dimensions[0]; i++)
    {
        for (int j = 0; j < dimensions[1]; j++)
        {
            I[0][i][j] = rand() % 256;
        }
    }
    for (int i = 0; i < dimensions[0]; i++)
    {
        for (int j = 0; j < dimensions[1]; j++)
        {
            J[0][i][j] = rand() % 256;
        }
    }

    // finding dimensions for all layers
    for (size_t L = 1; L <= Lm; L++)
    {
        // L-th [nx ny]: [floor((nx+1)/2) floor((ny+1)/2)]
        dimensions[2*L] = std::floor( (dimensions[2*(L-1)] + 1) / 2 );
        dimensions[2*L + 1] = std::floor( (dimensions[2*(L-1) + 1] + 1) / 2 );
    }


    // building pyramid layers
    // second variables in loops are for edge cases (2x-1 < 0,...)
    for (int L = 1; L <= Lm; L++)
    {
        I[L] = std::vector<std::vector<int>>(dimensions[2 * L], std::vector<int>(dimensions[2 * L + 1]));

        for (int y = 1, i = 0; y < dimensions[2 * L] - 1; y++, i++)
        {
            for (int x = 1, j = 0; x < dimensions[2 * L + 1] - 1; x++, j++)
            {
                I[L][i][j] = 1/4 * I[L-1][2*x][2*y] +
                             1/8 * (I[L-1][2*x - 1][2*y] + I[L-1][2*x + 1][2*y] + I[L-1][2*x][2*y - 1] + I[L-1][2*x][2*y + 1]) +
                             1/16 * (I[L-1][2*x - 1][2*y - 1] + I[L-1][2*x + 1][2*y + 1] + I[L-1][2*x - 1][2*y + 1] + I[L-1][2*x + 1][2*y + 1]);
                
                J[L][i][j] = 1/4 * J[L-1][2*x][2*y] +
                             1/8 * (J[L-1][2*x - 1][2*y] + J[L-1][2*x + 1][2*y] + J[L-1][2*x][2*y - 1] + J[L-1][2*x][2*y + 1]) +
                             1/16 * (J[L-1][2*x - 1][2*y - 1] + J[L-1][2*x + 1][2*y + 1] + J[L-1][2*x - 1][2*y + 1] + J[L-1][2*x + 1][2*y + 1]);
            }
        }
    }

    std::vector<std::vector<int>> g(Lm, std::vector<int>(2)); // pyramidal guess [0, 0]
    std::vector<int> u((Lm+1) * 2); // vector for point u for all layers
    std::vector<int> v(2); // vector for point v
    std::vector<std::vector<int>> bk(2, std::vector<int>(1)); // image mismatch vector
    std::vector<std::vector<int>> etak(2, std::vector<int>(1)); // guess for next iteration
    std::vector<std::vector<int>> d(Lm, std::vector<int>(2)); // final optical flow

    for (int L = Lm; L >= 0; L--)
    {
        // finding u
        u[2*L] = (u[2*L] + 1) / std::pow(2, L);
        u[2*L + 1] = (u[2*L + 1] + 1) / std::pow(2, L);

        // derivative Ix, Iy
        std::vector<std::vector<int>> Ix(dimensions[2*L],
                                     std::vector<int>(dimensions[2*L + 1]));
        std::vector<std::vector<int>> Iy(dimensions[2*L],
                                     std::vector<int>(dimensions[2*L + 1]));
        for (int y = 1, i = 0; y < dimensions[2*L] - 1; y++, i++)
        {
            for (int x = 1, j = 0; x < dimensions[2*L + 1] - 1; x++, j++)
            {
               Ix[i][j] = (I[L][x+1][y] - I[L][x-1][y]) / 2;
               Iy[i][j] = (I[L][x][y+1] - I[L][x][y-1]) / 2;
            }
        }

        // patial gradient matrix
        std::vector<std::vector<int>> G(2, std::vector<int>(2));
        for (int y = u[2*L + 1] - omega_y; y <= u[2*L + 1] + omega_y; y++)
        {
            for (int x = u[2*L] - omega_x; x <= u[2*L] + omega_x; x++)
            {
                G[0][0] += std::pow(Ix[x][y], 2);
                G[0][1] += Ix[x][y] * Iy[x][y];
                G[1][0] += Ix[x][y] * Iy[x][y];
                G[1][1] += std::pow(Iy[x][y], 2);
            }
        }
        
        std::vector<std::vector<int>> deltaIk;

        // initialization of iterative L-K
        std::vector<std::vector<int>> nu(K, std::vector<int>(2)); //  [0, 0]

        for (int k = 1; k <= K; k++)
        {
            // image difference
            for (int y = 1, i = 0; y < dimensions[2 * L] - 1; y++, i++)
            {
                for (int x = 1, j = 0; x < dimensions[2 * L + 1] - 1; x++, j++)
                {
                    deltaIk[x][y] = I[L][x][y] - J[L][x + g[L][0] + nu[k-1][0]][y + g[L][1] + nu[k-1][1]];
                }
            }

            // image mismatch vector
            for (int y = u[2*L + 1] - omega_y; y <= u[2*L + 1] + omega_y; y++)
            {
                for (int x = u[2*L] - omega_x; x <= u[2*L] + omega_x; x++)
                {
                    bk[0][0] += deltaIk[x][y] * Ix[x][y];
                    bk[0][0] += deltaIk[x][y] * Iy[x][y];
                }
            }

            // optical flow
            // etak = matrix_multiply(matrix_inverse(G), bk); // matrix functions not yet implemented

            // guess for next iteration
            nu[k][0] = nu[k-1][0] + etak[0][0];
            nu[k][1] = nu[k-1][1] + etak[1][0];
                        
        }

        // final optical flow at level L
        d[L][0] = nu[K][0];
        d[L][1] = nu[K][1];

        // guess for next level L-1
        g[L-1][0] = 2 * (g[L][0] + d[L][0]);
        g[L-1][1] = 2 * (g[L][1] + d[L][1]);

    }

    // final optical flow vector
    std::vector<int> d_final(2);
    d_final[0] = g[0][0] + d[0][0];
    d_final[1] = g[0][1] + d[0][1];

    // location of point on J
    v[0] = u[0] + d[0][0];
    v[1] = u[1] + d[0][1];


    return 0;
}