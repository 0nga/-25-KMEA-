package nl.uu.trafficmas.norm;

import java.util.Map;

import nl.uu.trafficmas.roadnetwork.Sensor;
import nl.uu.trafficmas.simulationmodel.AgentData;



public class SimpleMaxSpeedNormInstantiation extends NormInstantiation {

	public SimpleMaxSpeedNormInstantiation(NormScheme ns, String agentID) {
		super(ns, agentID);
	}
	
	@Override
	public boolean deadline(Map<String, AgentData> currentOrgKnowledge, int currentTime) {
		Sensor s = this.ns.sensorList.get(0);
		if(((SimpleMaxSpeedNormScheme)this.ns).agentNorms.contains(this.agentID))
		{
			((SimpleMaxSpeedNormScheme)this.ns).agentNorms.remove(this.agentID);
			return true;
		}
		return false;
	}
}
