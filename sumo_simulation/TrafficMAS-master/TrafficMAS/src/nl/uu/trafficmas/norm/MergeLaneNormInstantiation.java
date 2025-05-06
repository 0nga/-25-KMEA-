package nl.uu.trafficmas.norm;

import java.util.ArrayList;
import java.util.List;

import nl.uu.trafficmas.simulationmodel.AgentData;

public class MergeLaneNormInstantiation extends NormInstantiation {

	private List<AgentData> goals;
	
	public MergeLaneNormInstantiation(NormScheme ns, String agentID) {
		super(ns, agentID);
		goals = new ArrayList<AgentData>();
	}

	public void setLaneIndex(int laneIndex){
		goals.add(new AgentData(null, null, -1, -1, null, laneIndex, -1, -1));
	}
	
	@Override
	public List<AgentData> getGoals() {
		return goals;
	}
}
