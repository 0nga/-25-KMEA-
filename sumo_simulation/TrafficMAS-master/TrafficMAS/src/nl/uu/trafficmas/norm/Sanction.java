package nl.uu.trafficmas.norm;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;

import nl.uu.trafficmas.organisation.InstitutionalState;
import nl.uu.trafficmas.simulationmodel.AgentData;

public class Sanction implements InstitutionalState {
	public final SanctionType sanctionType;
	public final String agentID;
	
	public Sanction(SanctionType sanctionType, String agentID) {
		this.sanctionType = sanctionType;
		this.agentID	= agentID;
	}
	
	public static List<Entry<Sanction, Double>> getSanctionsAndDistance(AgentData agentData, List<NormInstantiation> normInst, int currentTime, int expectedExitTime) {
		List<Entry<Sanction, Double>> sanctionsAndDistance = new ArrayList<Entry<Sanction, Double>>();
		for(NormInstantiation ni : normInst) {
			List<AgentData> normGoals = ni.getGoals();
			double minDist = Double.POSITIVE_INFINITY;
			for(AgentData goal : normGoals) {
				double dist = AgentData.distance(agentData,goal, agentData.acceleration, agentData.deceleration);
				if(dist < minDist) {
					minDist = dist;
				}
					
			}
			if(minDist > 0) {
				Sanction s = ni.getSanction(agentData.id);
				double delta = calculateDelta(minDist, currentTime, expectedExitTime);
				Entry<Sanction,Double> e = new AbstractMap.SimpleEntry<Sanction, Double>(s, delta);
				sanctionsAndDistance.add(e);
			}
		}
		return sanctionsAndDistance;
	}
	
	public static double calculateDelta(double distance, int currentTime, int expectedExitTime) {
		int timeLeft = expectedExitTime - currentTime;
		return Math.min(distance, timeLeft)/(timeLeft);
	}
}
