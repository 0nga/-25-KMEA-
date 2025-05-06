package nl.uu.trafficmas.agent;

import nl.uu.trafficmas.roadnetwork.Lane;
import nl.uu.trafficmas.roadnetwork.Road;

public class PhysicalObject {
	protected Road road;
	protected Lane lane;
	protected double distance;
	
	public Road getRoad() {
		return road;
	}
	
	public void setRoad(Road road) {
		this.road = road;
	}
	public Lane getLane() {
		return lane;
	}
	
	public void setLane(Lane lane) {
		this.lane = lane;
	}
	public double getDistance() {
		return distance;
	}
	public void setDistance(double distance) {
		this.distance = distance;
	}

	
	
}
