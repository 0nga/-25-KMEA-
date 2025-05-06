package nl.uu.trafficmas.agent;

import nl.uu.trafficmas.roadnetwork.Node;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Route;
import de.tudresden.ws.container.SumoColor;

public class NormalAgent extends BasicAgent implements ISumoColor {

	public NormalAgent(String agentID, Node goalNode, Route route, RoadNetwork roadNetwork, int goalArrivalTime, double maxSpeed, double maxComfySpeed) {
		super(agentID, goalNode, route, roadNetwork, goalArrivalTime, maxSpeed, maxComfySpeed);
	}
	

	@Override
	public SumoColor getColor() {
		//red
		return new SumoColor(255,0,0,255);
	}
}
