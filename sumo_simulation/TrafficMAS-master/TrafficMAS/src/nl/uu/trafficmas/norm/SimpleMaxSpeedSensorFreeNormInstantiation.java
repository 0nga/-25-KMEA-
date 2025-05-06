package nl.uu.trafficmas.norm;

import java.util.Map;

import nl.uu.trafficmas.roadnetwork.Lane;
import nl.uu.trafficmas.roadnetwork.Sensor;
import nl.uu.trafficmas.simulationmodel.AgentData;



public class SimpleMaxSpeedSensorFreeNormInstantiation extends NormInstantiation {

	public SimpleMaxSpeedSensorFreeNormInstantiation(NormScheme ns, String agentID) {
		super(ns, agentID);
	}
	
	@Override
	public boolean deadline(Map<String, AgentData> currentOrgKnowledge, int currentTime) {
	
		// TODO: here something for checking if the agent is on the correct lane?
		AgentData ad = currentOrgKnowledge.get(this.agentID);
		
		if(ad == null)
			return false;
		
		SimpleMaxSpeedSensorFreeNormScheme castedNS = (SimpleMaxSpeedSensorFreeNormScheme)this.ns;
		
		boolean normActive = castedNS.agentNorms.contains(this.agentID);
		boolean isOnDeadlineRoad = castedNS.getDeadlineRoad().equals(ad.roadID);
		if(normActive && isOnDeadlineRoad)
		{
			castedNS.agentNorms.remove(this.agentID);
			return true;
		}
		return false;
	}
	
	@Override
	public boolean violated(AgentData ad) {
		// violated if you're not on the second lane
		SimpleMaxSpeedSensorFreeNormScheme castedNS = (SimpleMaxSpeedSensorFreeNormScheme)this.ns;
		Lane l = rn.getRoadFromID(ad.roadID).getLanes()[ad.laneIndex];
		return l.getID().equals(castedNS.getSecondLaneId());
			
	}
}
