// g++ -std=c++11 calc.cpp -o calc

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>

int main() {
    std::ifstream in("events.txt");
    std::ofstream out("events_with_4vec.txt");

    std::string line;

    while (std::getline(in, line)) {
        // Pass through event headers / separators
        if (line.rfind("Event", 0) == 0 || line == "----") {
            out << line << "\n";
            continue;
        }

        std::istringstream iss(line);

        float pt, eta, phi, mass;
        int id;

        if (!(iss >> pt >> eta >> phi >> mass >> id)) continue;

        // Compute derived quantities
        float px = pt * std::cos(phi);
        float py = pt * std::sin(phi);
        float pz = pt * std::sinh(eta);
        float p2 = px*px + py*py + pz*pz;
        float E  = std::sqrt(p2 + mass*mass);

        // Write everything out
        out << pt << " "
            << eta << " "
            << phi << " "
            << mass << " "
            << id << " "
            << px << " "
            << py << " "
            << pz << " "
            << E  << "\n";
    }

    in.close();
    out.close();

    return 0;
}
