/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 200;
	default_random_engine generator;

  normal_distribution<double> distX(x, std[0]);
  normal_distribution<double> distY(y,std[1]);
  normal_distribution<double> distTheta(theta,std[2]);

	for(int i=0;i<num_particles;++i){
		Particle p;
		p.id = i;
		p.x = distX(generator);
		p.y = distY(generator);
		p.theta = distTheta(generator);
		p.weight = 1;
		particles.push_back(p);
		weights.push_back(1.0);
	}

		is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine generator;

	for(int i=0; i<num_particles; ++i){

		normal_distribution<double> distX(0, std_pos[0]);
		normal_distribution<double> distY(0,std_pos[1]);
		normal_distribution<double> distTheta(0,std_pos[2]);

		if(fabs(yaw_rate) > 0.01){
			particles[i].x += (velocity/yaw_rate)*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			particles[i].y += (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));

		}
		else{
			particles[i].x += velocity*delta_t*cos(particles[i].theta);
			particles[i].y += velocity*delta_t*sin(particles[i].theta);
		}
		particles[i].x += distX(generator);
		particles[i].y += distY(generator);
		particles[i].theta = particles[i].theta + yaw_rate * delta_t + distTheta(generator);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	weights.clear();

	double std_x = std_landmark[0];
	double std_y = std_landmark[1];
	double normalizer = 1/(2.0*M_PI*std_x*std_y);

	for(int i=0;i<num_particles;++i){

		vector<LandmarkObs> landmarksInRange;

		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		for(int j=0;j<map_landmarks.landmark_list.size();++j){

			double distance = dist(map_landmarks.landmark_list[j].x_f , map_landmarks.landmark_list[j].y_f , x , y);
			if(distance <= sensor_range){
				LandmarkObs temp;
				temp.x = map_landmarks.landmark_list[j].x_f;
				temp.y = map_landmarks.landmark_list[j].y_f;
				temp.id = map_landmarks.landmark_list[j].id_i;
				landmarksInRange.push_back(temp);
			}
		}

		std::vector<LandmarkObs> predicted;
		std::vector<LandmarkObs> NearestObs;

		for(int j=0;j<observations.size();j++){

			double predict_x,predict_y;
			
			predict_x = x + observations[j].x * cos(theta) - observations[j].y * sin(theta);
			predict_y = y + observations[j].x * sin(theta) + observations[j].y * cos(theta);

			double min_dist = INFINITY;
			double mean_x, mean_y;
			int id=0;
			mean_x = 0.0;
			mean_y = 0.0;
			for (int m = 0; m < landmarksInRange.size(); ++m) {
				double dist_landmark = dist(predict_x, predict_y, landmarksInRange[m].x, landmarksInRange[m].y);
				if (dist_landmark < min_dist) {
					mean_x = landmarksInRange[m].x;
					mean_y = landmarksInRange[m].y;
					id = landmarksInRange[m].id;
					min_dist = dist_landmark;
				}
			}

			LandmarkObs temp;
			temp.x = predict_x;
			temp.y = predict_y;
			temp.id = id;

			predicted.push_back(temp);

			LandmarkObs nearest_temp;
			nearest_temp.x = mean_x;
			nearest_temp.y = mean_y;

			NearestObs.push_back(nearest_temp);
		}

		double update_weight = 1;

		for(int w = 0; w < predicted.size(); ++w){
			double dx = predicted[w].x - NearestObs[w].x;
			double dy = predicted[w].y - NearestObs[w].y;
			update_weight *= normalizer*exp(-(dx*dx/(2*std_x*std_x)) - (dy*dy/(2*std_y*std_y)));
		}
		particles[i].weight = update_weight;
		weights.push_back(particles[i].weight);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine generator;
	vector<Particle> resampled_particles;

	discrete_distribution<int> weights_distribution(weights.begin(),weights.end());
	for(int i=0; i<num_particles; ++i){
		resampled_particles.push_back(particles[weights_distribution(generator)]);
	}
	particles = resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

		return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
