package nl.uu.trafficmas.norm;

import java.util.Map;

import nl.uu.trafficmas.roadnetwork.Sensor;
import nl.uu.trafficmas.simulationmodel.AgentData;



public class MaxSpeedNormInstantiation extends NormInstantiation {

	public MaxSpeedNormInstantiation(NormScheme ns, String agentID) {
		super(ns, agentID);
	}
	
	@Override
	public boolean deadline(Map<String, AgentData> currentOrgKnowledge, int currentTime) {
		MaxSpeedNormScheme castedScheme = (MaxSpeedNormScheme)this.ns;
		if(currentOrgKnowledge.get(this.agentID) == null) {
			return false;
		}
		boolean atDeadLineRoad = currentOrgKnowledge.get(this.agentID).roadID.equals(castedScheme.deadlineAtRoad);
		if(castedScheme.agentNorms.contains(this.agentID) && 
				atDeadLineRoad)
		{
			castedScheme.agentNorms.remove(this.agentID);
			return true;
		}
		return false;
	}
}
