package nl.uu.trafficmas.datamodel;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import nl.uu.trafficmas.agent.AgentProfileType;
import nl.uu.trafficmas.organisation.CommunicationHub;
import nl.uu.trafficmas.organisation.Organisation;

public class MASData {
	public final int simulationLength;
	public final LinkedHashMap<String, Double> spawnProbabilities;
	public final boolean multipleRoutes;
	public final HashMap<String, LinkedHashMap<AgentProfileType, Double>> routeAgentTypeSpawnDist;
	public final Map<String, Organisation> organisations;
	public final double rightLaneRatio;
	public final CommunicationHub ch;
	public MASData(int simulationLength, 
			LinkedHashMap<String,Double> spawnProbabilities, 
			double rightLaneRatio,
			boolean multipleRoutes,
			HashMap<String, LinkedHashMap<AgentProfileType, Double>> routeAgentTypeSpawnDist, 
			Map<String, Organisation> organisations,CommunicationHub ch) {
		this.simulationLength 			= simulationLength;
		this.spawnProbabilities 		= spawnProbabilities;
		this.rightLaneRatio				= rightLaneRatio;
		this.multipleRoutes 			= multipleRoutes;
		this.routeAgentTypeSpawnDist 	= routeAgentTypeSpawnDist;
		this.organisations				= organisations;
		this.ch 						= ch;
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("SimulationLength: "+simulationLength+"\r\n");
		sb.append("spawnProbability: "+spawnProbabilities+"\r\n");
		sb.append("rightLaneRatio: "+rightLaneRatio+"\r\n");
		sb.append("multipleRoutes: "+multipleRoutes+"\r\n");
		sb.append("AgentType Spawnprob per route"+routeAgentTypeSpawnDist+"\r\n");
		return sb.toString();
	}
	
}
