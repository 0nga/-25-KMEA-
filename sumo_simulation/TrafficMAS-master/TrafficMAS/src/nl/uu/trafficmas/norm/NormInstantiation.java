package nl.uu.trafficmas.norm;

import java.util.List;
import java.util.Map;

import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.simulationmodel.AgentData;

public abstract class NormInstantiation {
	protected final NormScheme ns;
	public final String agentID;
	protected RoadNetwork rn;
	
	public NormInstantiation(NormScheme ns, String agentID) {
		this.ns = ns;
		this.agentID = agentID;
	}
	public String agentID() {
		return agentID;
	}

	public boolean violated(AgentData ad) {
		return ns.violated(ad);
	};

	public void addRoadNetwork(RoadNetwork rn) {
		this.rn = rn;
	}
	
	public Sanction getSanction(String agentID) {
		return ns.getSanction(agentID);
	}

	public boolean deadline(Map<String, AgentData> currentOrgKnowledge, int currentTime) {
		return ns.deadline(currentOrgKnowledge, currentTime);
	}
	
	@Override
	public boolean equals(Object ni)  {
		if(!(ni instanceof NormInstantiation))
			return false;
		return ((NormInstantiation) ni).ns.id.equals(this.ns.id) && ((NormInstantiation) ni).agentID.equals(this.agentID);	
	}
	
	@Override
	public int hashCode() {
		return (int) (Integer.valueOf(this.agentID.charAt(0))+(Integer.valueOf(this.ns.id.charAt(0))));
	}
	public List<AgentData> getGoals() {
		// TODO Auto-generated method stub
		return ns.getGoals();
	}
}
