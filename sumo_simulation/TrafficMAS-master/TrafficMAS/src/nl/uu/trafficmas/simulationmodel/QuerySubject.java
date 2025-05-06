package nl.uu.trafficmas.simulationmodel;

import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;

import nl.uu.trafficmas.agent.Agent;
import de.tudresden.sumo.util.SumoCommand;
import de.tudresden.ws.container.SumoStringList;

public enum QuerySubject {
	Vehicle,
	Lane,
	Edge, Sensor;

	public SumoCommand getIDCountCommand() {
		SumoCommand cmd = null;
		switch(this) {
		case Edge:
			cmd = de.tudresden.sumo.cmd.Edge.getIDCount();
			break;
		case Lane:
			cmd = de.tudresden.sumo.cmd.Lane.getIDCount();
			break;
		case Vehicle:
			cmd = de.tudresden.sumo.cmd.Vehicle.getIDCount();
			break;
		case Sensor:
			cmd = de.tudresden.sumo.cmd.ArealDetector.getIDCount();
			break;
		}
		return cmd;
	}

	public SumoCommand getIDListCommand() {
		SumoCommand cmd = null;
		switch(this) {
		case Edge:
			cmd = de.tudresden.sumo.cmd.Edge.getIDList();
			break;
		case Lane:
			cmd = de.tudresden.sumo.cmd.Lane.getIDList();
			break;
		case Vehicle:
			cmd = de.tudresden.sumo.cmd.Vehicle.getIDList();
			break;
		case Sensor:
			cmd = de.tudresden.sumo.cmd.ArealDetector.getIDList();
			break;
		}
		return cmd;
	}

	public SumoCommand queryFieldToCommand(QueryField queryField, String id) throws Exception {
		SumoCommand cmd = null;
		switch(this) {
		case Edge:
			switch(queryField) {
			case MeanSpeed:
				cmd = de.tudresden.sumo.cmd.Edge.getLastStepMeanSpeed(id);
				break;
			case MeanTime:
				cmd = de.tudresden.sumo.cmd.Edge.getTraveltime(id);
				break;
			default:
				throw new Exception(this+" cannot handle " + queryField);
			}
			break;
		case Lane:
			switch(queryField) {
			case MeanSpeed:
				cmd = de.tudresden.sumo.cmd.Lane.getLastStepMeanSpeed(id);
				break;
			case MeanTime:
				cmd = de.tudresden.sumo.cmd.Lane.getTraveltime(id);
				break;
			case EdgeId:
				cmd = de.tudresden.sumo.cmd.Lane.getEdgeID(id);
				break;
			case LaneLength:
				cmd = de.tudresden.sumo.cmd.Lane.getLength(id);
				break;
			default:
				throw new Exception(this+" cannot handle " + queryField);
			}
			break;
		case Vehicle:
			switch(queryField) {
			case LeadingVehicle:
				//TODO: make distance (100) vehicle type dependent?
				cmd = de.tudresden.sumo.cmd.Vehicle.getLeader(id, 100);
				break;
			case Position:
				cmd = de.tudresden.sumo.cmd.Vehicle.getLanePosition(id);
				break;
			case Speed:
				cmd = de.tudresden.sumo.cmd.Vehicle.getSpeed(id);
				break;
			case EdgeId:
				cmd = de.tudresden.sumo.cmd.Vehicle.getRoadID(id);
				break;
			case LaneIndex:
				cmd = de.tudresden.sumo.cmd.Vehicle.getLaneIndex(id);
				break;
			default:
				throw new Exception(this+" cannot handle " + queryField);
			}
			break;
		case Sensor:
			switch(queryField) {
			case VehicleIDList:
				cmd = de.tudresden.sumo.cmd.ArealDetector.getLastStepVehicleIDs(id);
				break;
			case Speed:
				cmd = de.tudresden.sumo.cmd.ArealDetector.getLastStepMeanSpeed(id);
				break;
			default:
				throw new Exception(this+" cannot handle " + queryField);
			}
		}
		return cmd;
	}

	public Data toData(LinkedHashSet<QueryField> linkedHashSet,
			List<Object> subList, String id) {
		Data data = null;
		String [] 	edgeId 				= new String[1];
		int	   [] 	laneIndex			= new int[1];
		Object [][] leadingVehicle 		= new Object[1][2];
		double [] 	meanSpeed 			= new double[1];
		double [] 	meanTime 			= new double[1];
		double [] 	position 			= new double[1];
		double [] 	speed				= new double[1];
		double [] 	length				= new double[1];
		SumoStringList [] vehicleList	= new SumoStringList[1];
		Iterator<QueryField> queryFieldIt = linkedHashSet.iterator();
		Iterator<Object> responseIt = subList.iterator();
		parseResponses(edgeId, leadingVehicle, meanSpeed, meanTime,
				position, speed, laneIndex,vehicleList, length, queryFieldIt,responseIt);
		switch (this) {
		case Edge:
			data = new EdgeData(id, meanSpeed[0],meanTime[0]);
			break;
		case Lane:
			data = new LaneData(id, meanSpeed[0], meanTime[0], edgeId[0], length[0]);
			break;
		case Vehicle:
			data = new AgentData(id, leadingVehicle[0], position[0], speed[0],edgeId[0],laneIndex[0],Agent.deceleration, Agent.acceleration);
			break;
		case Sensor:
			String [] vehicleIds = new String[vehicleList[0].size()];
			vehicleList[0].toArray(vehicleIds);
			data = new SensorData(id,vehicleIds,speed[0]);
			break;
		}
		
		return data;
	}

	private void parseResponses(String[] edgeId, Object[][] leadingVehicle,
			double[] meanSpeed, double[] meanTime, double[] position,
			double[] speed, int [] laneIndex, SumoStringList [] vehicleList, double[] length,
			Iterator<QueryField> queryFieldIt, Iterator<Object> responseIt) {
		while(queryFieldIt.hasNext() && responseIt.hasNext()) {
			QueryField queryField = queryFieldIt.next();
			Object response = responseIt.next();
			switch(queryField) {
			case EdgeId:
				edgeId[0] = (String) response;
				break;
			case LeadingVehicle:
				leadingVehicle[0] = (Object[]) response;
				break;
			case MeanSpeed:
				meanSpeed[0] = (double) response;
				break;
			case MeanTime:
				meanTime[0] = (double) response;
				break;
			case Position:
				position[0] = (double) response;
				break;
			case Speed:
				speed[0] = (double) response;
				break;
			case LaneIndex:
				laneIndex[0] = (int) response;
				break;
			case VehicleIDList:
				vehicleList[0] = (SumoStringList) response;
				break;
			case LaneLength:
				length[0] = (double) response;
				break;
			default:
				break;
			}
		}
	}
}
