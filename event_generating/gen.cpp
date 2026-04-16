// g++ -std=c++11 gen.cpp -o gen

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>

// Simple particle struct
struct Particle {
    float pt;
    float eta;
    float phi;
    float mass;
    int id; // 0=photon, 1=charged hadron, 2=neutral hadron
};

// ---------- helper to inject a simple resonance dijet ----------
std::array<Particle,2> resonance_dijet(float M, std::mt19937 &gen) {
    std::uniform_real_distribution<float> uni(-1.0,1.0);
    std::uniform_real_distribution<float> phi_dist(-M_PI,M_PI);
    std::uniform_int_distribution<int> pid_dist(0,2); // particle ID

    float theta = std::acos(uni(gen));
    float phi = phi_dist(gen);
    float p = M/2.0;  // assume massless daughters

    float px = p * std::sin(theta) * std::cos(phi);
    float py = p * std::sin(theta) * std::sin(phi);
    float pz = p * std::cos(theta);

    float pt = std::sqrt(px*px + py*py);
    float eta = 0.5 * std::log((p + pz)/(p - pz + 1e-8));
    float mass = 0.0;

    int pid1 = pid_dist(gen);
    int pid2 = pid_dist(gen);

    Particle p1 = {pt, eta, static_cast<float>(phi), mass, pid1};
    Particle p2 = {pt, -eta, static_cast<float>(phi + M_PI), mass, pid2};

    return {p1,p2};
}

// Generate one event
std::vector<Particle> generate_event(int n_particles, std::mt19937 &gen) {
    std::vector<Particle> event;
    event.reserve(n_particles);

    // Distributions
    std::uniform_real_distribution<float> eta_dist(-2.5, 2.5);
    std::uniform_real_distribution<float> phi_dist(-M_PI, M_PI);
    std::uniform_real_distribution<float> uniform01(0.0, 1.0);
    std::uniform_int_distribution<int> id_dist(0, 2);

    // Exponential pT: pT = -lambda * log(1 - u)
    float lambda = 50.0; // controls falloff

    // ---------- background particles ----------
    for (int i = 0; i < n_particles-2; i++) {  // leave space for resonance
        float u = uniform01(gen);
        float pt = -lambda * std::log(1.0 - u);

        float eta = eta_dist(gen);
        float phi = phi_dist(gen);

        int id = id_dist(gen);

        float mass;
        if (id == 0) mass = 0.0;       // photon
        else if (id == 1) mass = 0.14; // charged hadron
        else mass = 0.5;               // neutral hadron

        event.push_back({pt, eta, phi, mass, id});
    }

    // ---------- inject resonance dijet ----------
    auto res = resonance_dijet(200.0, gen); // 200 GeV resonance
    event.push_back(res[0]);
    event.push_back(res[1]);

    // ---------- optional smearing ----------
    std::normal_distribution<float> smear(0.0,5.0); // sigma=5 GeV
    for (auto &p : event) p.pt += smear(gen);

    return event;
}

int main() {
    std::ofstream out("events.txt");

    std::mt19937 gen(42); // fixed seed for reproducibility

    int n_events = 5;
    int particles_per_event = 100;

    for (int i = 0; i < n_events; i++) {
        auto event = generate_event(particles_per_event, gen);

        out << "Event " << i << "\n";
        for (const auto &p : event) {
            out << p.pt << " "
                      << p.eta << " "
                      << p.phi << " "
                      << p.mass << " "
                      << p.id << "\n";
        }
        out << "----\n";
    }

    out.close();

    return 0;
}
