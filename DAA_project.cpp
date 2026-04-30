// aco_tsp.cpp
// Simple Ant Colony Optimization for small TSP (suitable for DAA project)
// Compile: g++ -std=c++11 aco_tsp.cpp -O2 -o aco
// Run: ./aco

#include <bits/stdc++.h>
using namespace std;

struct Point { double x, y; };
double euclid(const Point &a, const Point &b){
    double dx = a.x - b.x, dy = a.y - b.y; return sqrt(dx*dx + dy*dy);
}

int main(){
    // --- PARAMETERS (tune these for experiments) ---
    int nNodes = 10;            // number of cities (you can change & test)
    int nAnts = 20;
    int maxIter = 200;
    double alpha = 1.0;        // pheromone importance
    double beta = 5.0;         // heuristic importance
    double rho = 0.5;          // evaporation rate (0..1)
    double Q = 100.0;          // pheromone deposit factor
    unsigned seed = 12345;
    mt19937 rng(seed);

    // --- Sample coordinates (you can replace by file input) ---
    vector<Point> pts = {
        {0,0},{1,5},{5,2},{3,3},{8,8},{7,2},{6,6},{2,8},{9,1},{4,7}
    };
    if ((int)pts.size() < nNodes) {
        // If you changed nNodes, auto-fill random points
        uniform_real_distribution<double> dist(0, 10.0);
        while ((int)pts.size() < nNodes) pts.push_back({dist(rng), dist(rng)});
    } else {
        nNodes = (int)pts.size();
    }

    // --- Distance & heuristic matrices ---
    vector<vector<double>> dist(nNodes, vector<double>(nNodes));
    for (int i=0;i<nNodes;i++) for (int j=0;j<nNodes;j++)
        dist[i][j] = (i==j? 1e9 : euclid(pts[i], pts[j]));

    vector<vector<double>> eta(nNodes, vector<double>(nNodes)); // 1/d
    for (int i=0;i<nNodes;i++) for (int j=0;j<nNodes;j++)
        eta[i][j] = (i==j? 0.0 : 1.0 / dist[i][j]);

    // --- Pheromone matrix ---
    double tau0 = 1.0; // initial pheromone
    vector<vector<double>> tau(nNodes, vector<double>(nNodes, tau0));
    vector<int> bestTour; double bestLen = 1e18;

    // Helper: compute tour length
    auto tour_length = [&](const vector<int> &tour){
        double L=0;
        for (size_t i=0;i<tour.size()-1;i++) L += dist[tour[i]][tour[i+1]];
        L += dist[tour.back()][tour.front()];
        return L;
    };

    // Main loop
    for (int iter=0; iter<maxIter; ++iter){
        vector<vector<int>> antsTours(nAnts);
        vector<double> antsLen(nAnts, 0.0);

        for (int k=0;k<nAnts;k++){
            vector<int> visited(nNodes, 0);
            vector<int> tour; tour.reserve(nNodes);
            int start = k % nNodes; // spread starts
            tour.push_back(start); visited[start]=1;

            for (int step=1; step<nNodes; ++step){
                int cur = tour.back();
                // compute probabilities for unvisited nodes
                vector<double> probs(nNodes, 0.0);
                double sum=0;
                for (int j=0;j<nNodes;j++) if (!visited[j]){
                    probs[j] = pow(tau[cur][j], alpha) * pow(eta[cur][j], beta);
                    sum += probs[j];
                }
                // roulette wheel
                uniform_real_distribution<double> ud(0.0, 1.0);
                if (sum <= 0){
                    // fallback: pick random unvisited
                    vector<int> freev;
                    for (int j=0;j<nNodes;j++) if (!visited[j]) freev.push_back(j);
                    uniform_int_distribution<int> ui(0, (int)freev.size()-1);
                    int pick = freev[ui(rng)];
                    tour.push_back(pick); visited[pick]=1;
                } else {
                    double r = ud(rng) * sum;
                    double acc=0; int chosen=-1;
                    for (int j=0;j<nNodes;j++) if (!visited[j]){
                        acc += probs[j];
                        if (acc >= r){ chosen = j; break; }
                    }
                    if (chosen == -1){ // numerical issues
                        for (int j=0;j<nNodes;j++) if (!visited[j]) { chosen = j; break; }
                    }
                    tour.push_back(chosen); visited[chosen]=1;
                }
            }
            antsTours[k] = tour;
            antsLen[k] = tour_length(tour);
            if (antsLen[k] < bestLen){
                bestLen = antsLen[k]; bestTour = antsTours[k];
            }
        }

        // optional: local search like 2-opt could be applied here for improvement (omitted for brevity)

        // Pheromone evaporation
        for (int i=0;i<nNodes;i++) for (int j=0;j<nNodes;j++)
            tau[i][j] *= (1.0 - rho);

        // Pheromone deposit by ants
        for (int k=0;k<nAnts;k++){
            double deposit = Q / antsLen[k];
            const auto &tour = antsTours[k];
            for (int i=0;i<nNodes;i++){
                int a = tour[i], b = tour[(i+1)%nNodes];
                tau[a][b] += deposit;
                tau[b][a] += deposit;
            }
        }

        // Optionally: keep pheromone on best-so-far extra
        double bestDeposit = Q / bestLen;
        for (int i=0;i<nNodes;i++){
            int a = bestTour[i], b = bestTour[(i+1)%nNodes];
            tau[a][b] += bestDeposit;
            tau[b][a] += bestDeposit;
        }

        // Logging per iteration (for convergence plot)
        if (iter % 10 == 0)
            cerr << "Iter " << iter << ": bestLen = " << bestLen << "\n";
    }

    // Output best tour
    cout << "Best tour length: " << bestLen << "\nBest tour order: ";
    for (int v: bestTour) cout << v << " ";
    cout << "\nCoordinates order:\n";
    for (int idx: bestTour) cout << idx << ": ("<<pts[idx].x<<","<<pts[idx].y<<")\n";

    return 0;
}