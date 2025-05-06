package nl.uu.trafficmas.norm;

import java.util.ArrayList;
import java.util.List;

import nl.uu.trafficmas.simulationmodel.AgentData;

public class StayInLaneNormInstantiation extends NormInstantiation {

	private List<AgentData> goals;
	
	public StayInLaneNormInstantiation(NormScheme ns, String agentID) {
		super(ns, agentID);
		goals = new ArrayList<AgentData>();
	}

	public void addGoal(AgentData agentData){
		goals.add(agentData);
	}
	
	@Override
	public List<AgentData> getGoals() {
		return goals;
	}
	
}
