package nl.uu.trafficmas.agent;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import nl.uu.trafficmas.norm.Sanction;
import nl.uu.trafficmas.norm.SanctionType;
import nl.uu.trafficmas.roadnetwork.Node;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Route;
import de.tudresden.ws.container.SumoColor;

public class SUMODefaultAgent extends Agent implements ISumoColor {

	public SUMODefaultAgent(String agentID, Node goalNode, Route route,
			RoadNetwork roadNetwork, int goalArrivalTime, double maxSpeed,
			double maxComfySpeed) {
		super(agentID, goalNode, route, roadNetwork, goalArrivalTime, maxSpeed,
				maxComfySpeed);
	}

	@Override
	public SumoColor getColor() {
		return new SumoColor(255,255,0,255);
	}

	@Override
	protected Map<SanctionType, Double> sanctionTypeIntensity(Set<SanctionType> sanctionsTypes,
			List<Sanction> currentSanctionList) {

		Map<SanctionType,Double> sanctionTypesIntensity = new HashMap<SanctionType,Double>();
		for(SanctionType st : sanctionsTypes) {
			sanctionTypesIntensity.put(st, 0.0);
		}
		return sanctionTypesIntensity;
	}

	@Override
	protected double calculateTimeUtility(double time) {
		return 0;
	}

}
