package nl.uu.trafficmas.roadnetwork;

import java.util.ArrayList;
import java.util.Map;

public class Route {
	public final String routeID;
	private ArrayList<Edge> route;
	
	public String getRouteID(){
		return routeID;
	}
	
	public Route(String routeID,ArrayList<Edge> route) {
		this.routeID = routeID;
		this.route = route;
	}
	
	public Edge[] getRoute() {
		Edge [] routeArray = new Edge[route.size()];
		return route.toArray(routeArray);
	}

	/**
	 * Gets the remaining length of the route *after*
	 * the current road
	 * @param route the route the remaining length is calculated about
	 * @param road the current road
	 * @return the remaining length of the route *after* the current road if the road is part
	 * of the route. Double.POSITIVE_INFINITY otherwise.
	 */
	public static double getRouteRemainderLength(Edge[] route, Road road) {
		double length = 0;
		boolean found = false;
		for(Edge edge : route) {
			
			// road found, start counting *after* this one.
			if(edge.getRoad().equals(road)) {
				found = true;
				continue;
			}
			if(found) {
				length += edge.getRoad().length;
			}
		}
		if(!found)
			length = Double.POSITIVE_INFINITY;

		return length;
	}
	
	public static double getRouteRemainderTime(Edge[] route, RoadNetwork roadNetwork, Road currentRoad,  double agentMaxCompfySpeed) {
		Map<String,Double>roadBestLaneAverageTime = roadNetwork.getRoadBestLaneAverageTravelTime();
		double expectedArrivalTime = 0;
		for(Edge edge : route) {
			if(edge.getRoad().equals(currentRoad)) {
				continue;
			}
			double roadTimeMaxSpeed = edge.getRoad().length/agentMaxCompfySpeed;
			expectedArrivalTime += Math.max(roadBestLaneAverageTime.get(edge.getID()),roadTimeMaxSpeed);
		}
		
		return expectedArrivalTime;
	}
	
}
