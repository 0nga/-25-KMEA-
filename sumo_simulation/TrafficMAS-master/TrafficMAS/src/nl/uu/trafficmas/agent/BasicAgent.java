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

public abstract class BasicAgent extends Agent {

	public BasicAgent(String agentID, Node goalNode, Route route,
			RoadNetwork roadNetwork, int goalArrivalTime, double maxSpeed,
			double maxComfySpeed) {
		super(agentID, goalNode, route, roadNetwork, goalArrivalTime, maxSpeed,
				maxComfySpeed);
	}

	@Override
	protected Map<SanctionType, Double> sanctionTypeIntensity(Set<SanctionType> sanctionsTypes,
			List<Sanction> currentSanctionList) {
		Map<SanctionType,Double> sanctionTypesIntensity = new HashMap<SanctionType,Double>();
		double intensity = 0;
		for(SanctionType st : sanctionsTypes) {
			switch(st) {
			case LowFine:
				intensity = .2;
				break;
			case HighFine:
				intensity = 2;
				break;	
			case InfiniteFine:
				intensity = Double.MAX_VALUE;
				break;
			}
			sanctionTypesIntensity.put(st, intensity);
		}
		
		return sanctionTypesIntensity;
	}


	@Override
	protected double calculateTimeUtility(double time) {

		double utility = (double) (this.getGoalArrivalTime() / (time));

		return utility;
	}
}
