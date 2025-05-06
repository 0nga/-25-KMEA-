package nl.uu.trafficmas.simulationmodel;

import it.polito.appeal.traci.SumoTraciConnection;

import java.io.IOException;
import java.net.Socket;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import nl.uu.trafficmas.agent.Agent;
import nl.uu.trafficmas.agent.ISumoColor;
import nl.uu.trafficmas.agent.SUMODefaultAgent;
import nl.uu.trafficmas.agent.actions.AgentAction;
import nl.uu.trafficmas.agent.actions.ChangeLaneAction;
import nl.uu.trafficmas.agent.actions.SumoAgentAction;
import nl.uu.trafficmas.roadnetwork.Edge;
import nl.uu.trafficmas.roadnetwork.Lane;
import nl.uu.trafficmas.roadnetwork.Road;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import de.tudresden.sumo.cmd.Simulation;
import de.tudresden.sumo.cmd.Vehicle;
import de.tudresden.sumo.util.SumoCommand;
import de.tudresden.ws.container.SumoColor;
import de.tudresden.ws.container.SumoStringList;

public class SimulationModelTraaS implements SimulationModel {

	SumoTraciConnection conn;
	String sumocfg;
	String sumoBin;
	private Socket sa;
	public static final int LOOK_AHEAD_DISTANCE = 100;
	public static final int OVERTAKE_DURATION = 5;
	
	public SimulationModelTraaS(String sumoBin, String sumocfg){
		this.sumoBin = sumoBin;
		this.sumocfg = sumocfg;
	}
	
	public SimulationModelTraaS(String address, int port) {
		// TODO Auto-generated constructor stub
		try {
			sa = new Socket(address, port);
			conn = new SumoTraciConnection(sa.getRemoteSocketAddress());
		} catch (IOException | InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	@Override
	public void initialize() {
		if(conn == null) {
			conn = initialize(sumoBin, sumocfg);
		}		
 	}	
	
	public static SumoTraciConnection initialize(String sumoBin, String sumocfg){
		SumoTraciConnection conn = new SumoTraciConnection(sumoBin, sumocfg);
		// Add an extra option.
		try {	
			//start TraCI
			conn.runServer();
	       //load routes and initialize the simulation
			conn.do_timestep();    
       }catch(Exception ex){ex.printStackTrace();}
		
		return conn;
	}
	
	@Override
	public void initializeWithOptions(HashMap<String,String> optionValueMap) {
		if(conn == null) {
			conn = initializeWithOptions(optionValueMap, sumoBin, sumocfg);
		}		
 	}		
	public static SumoTraciConnection initializeWithOptions(HashMap<String,String> optionValueMap, String sumoBin, String sumocfg){
		SumoTraciConnection conn = new SumoTraciConnection(sumoBin, sumocfg);
		// Add an extra option.
		for(Entry<String, String> keyValue : optionValueMap.entrySet()) {
			conn.addOption(keyValue.getKey(), keyValue.getValue());
		}
		try {	
			//start TraCI
			conn.runServer();
	       //load routes and initialize the simulation
			conn.do_timestep();    
       }catch(Exception ex){ex.printStackTrace();}
		
		return conn;
	}

	@Override
	public void close() {
		if(sa != null) {
			try {
				sa.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		conn.close();
	}
	
	@Override
	public void doTimeStep() {
		try {
			conn.do_timestep();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public void addAgent(Agent agent, String routeID, int tick) {
		addAgent(agent, routeID, tick, conn);
	}
	
	/**
	 * Calls the method SimulationModelTraas.addAgentCommand(agent,routeID,tick,0)
	 * @param agent
	 * @param routeID
	 * @param tick
	 * @param conn
	 */
	public static void addAgent(Agent agent, String routeID, int tick, SumoTraciConnection conn){
		try {
			conn.do_job_set(addAgentCommand(agent, routeID, tick, (byte) 0));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Returns the SumoCommand to add a single agent 'agent' at time 'tick' with route 'routeID' 
	 * and certain default values such as spawn speed and Vehicle Type.
	 * @param agent
	 * @param routeID
	 * @param tick
	 * @return a SumoCommand to add a single agent.
	 */
	public static SumoCommand addAgentCommand(Agent agent, String routeID, int tick, byte laneIndex) {
		return Vehicle.add(agent.agentID, "Car", routeID, tick, 0.0, Math.min(agent.getMaxComfySpeed(),3), laneIndex);
	}
	
	@Override
	public HashMap<String, Agent> addAgents(HashMap<Agent, Integer> agentPairList, Random rng, double rightLaneRatio) {
		return addAgents(agentPairList, rng, rightLaneRatio, conn);
	}
	
	/**
	 * Sends a list of commands to spawn Agents to SUMO via connection 'conn'. The agent data and spawn times are provided in 'agentPairList'. 
	 * Default values depending on the agentProfileType are added, like Color and MaxSpeed.
	 * @param agentPairList
	 * @param conn
	 * @return the HashMap containing all agents that will spawn in the simulation. 
	 */
	public static HashMap<String, Agent> addAgents(HashMap<Agent, Integer> agentPairList, Random rng, double rightLaneRatio, SumoTraciConnection conn){
		HashMap<String, Agent> completeAgentMap = new LinkedHashMap<String, Agent>();
		ArrayList<SumoCommand> cmds = new ArrayList<>();
		
		if(rightLaneRatio == -1){
			rightLaneRatio = 1;
		}
		
		try {
			for( Entry<Agent, Integer> agentPair : agentPairList.entrySet()){
				
				Agent agent = agentPair.getKey();
				
				// Determine a random lane for the agent to spawn on.
				Road r = agent.getRoadNetwork().getRoadFromID(agent.getRouteStringList().get(0));
				
				//todo distribute with 60% 40%
				
				// keep flipping coins until we get < 0.6 or until we run out of lanes.
				double coinFlip = rng.nextDouble();
				int randomLaneIndex = 0;
				while(coinFlip >= rightLaneRatio && randomLaneIndex+1 <= r.getLanes().length-1) {
					randomLaneIndex++;
					coinFlip = rng.nextDouble();
				}
				
				byte laneIndex = r.getLanes()[randomLaneIndex].laneIndex;
				
				// Use SUMO default settings for this agent.
				if(agent instanceof SUMODefaultAgent){
					cmds.add(addAgentCommand(agent, agent.getRouteID(), agentPair.getValue(), laneIndex));
					cmds.add(Vehicle.setColor(agent.agentID, ((ISumoColor) agent).getColor()));
					completeAgentMap.put(agent.agentID, agent);
				} else{
				// Otherwise for our agents, disable setSpeedMode, LaneChangeMode and set custom maxSpeed. 
					cmds.add(addAgentCommand(agent, agent.getRouteID(), agentPair.getValue(), laneIndex));
					cmds.add(Vehicle.setLaneChangeMode(agent.agentID, 0b1100000000));
					cmds.add(Vehicle.setSpeedMode(agent.agentID, 0b00001));
					cmds.add(Vehicle.setMaxSpeed(agent.agentID, agent.getMaxComfySpeed()));
					cmds.add(Vehicle.setColor(agent.agentID, ((ISumoColor) agent).getColor()));
					completeAgentMap.put(agent.agentID, agent);
				}
				// TODO: think of a way to express max comfy speed in a different way
			}
	
			if(agentPairList.size() > 0){
				conn.do_jobs_set(cmds);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return completeAgentMap;
	}
	
	@Override
	public HashMap<String, Agent> updateCurrentAgentMap(HashMap<String, Agent> completeAgentMap, HashMap<String, Agent> oldAgentMap) {
		return updateCurrentAgentMap(completeAgentMap, oldAgentMap, conn);
	}
	
	/**
	 * Updates 'oldAgentMap' to contain only the agents that are currently present in the SUMO application. 
	 * It uses 'completeAgentMap' to add agents which have just spawned to the map.
	 * @param completeAgentMap
	 * @param oldAgentMap
	 * @param conn
	 * @return a new HashMap currentAgentMap, which contains all agents currently in the simulation.
	 */
	public static HashMap<String, Agent> updateCurrentAgentMap(HashMap<String, Agent> completeAgentMap, HashMap<String, Agent> oldAgentMap, SumoTraciConnection conn){
		HashMap<String, Agent> currentAgentMap = new LinkedHashMap<String, Agent>(oldAgentMap);
		try {
			
			ArrayList<SumoCommand> cmdList = new ArrayList<>();
			cmdList.add(Simulation.getArrivedIDList());
			cmdList.add(Simulation.getDepartedIDList());
			LinkedList<Object> responses = conn.do_jobs_get(cmdList);
			
			SumoStringList departedVehicleIDList = (SumoStringList) responses.remove();
			SumoStringList arrivedVehicleIDList = (SumoStringList) responses.remove();			
			
			for(String departedVehicleID: departedVehicleIDList){
				currentAgentMap.remove(departedVehicleID);
			}

			for(String arrivedVehicleID: arrivedVehicleIDList){
				currentAgentMap.put(arrivedVehicleID, completeAgentMap.get(arrivedVehicleID));

			}
			
		} catch (Exception e){
			e.printStackTrace();
		}
		return currentAgentMap;
	}
	
	@Override
	public void simulateAgentActions(HashMap<Agent, AgentAction> actions) throws Exception {
		simulateAgentActions(actions, conn);
	}
	
	/**
	 * Sends a list of commands to SUMO. The commands are generated from the HashMap 'actions' and are all actions the vehicle can take.
	 * The following actions can be send to SUMO:
	 * 
	 * ChangeLane,
	 * ChangeRoad,
	 * ChangeVelocity5,
	 * ChangeVelocity10,
	 * ChangeVelocity20,
	 * ChangeVelocityMax
	 * @param actions
	 * @param conn
	 * @throws Exception 
	 */
	public static void simulateAgentActions(HashMap<Agent, AgentAction> actions, SumoTraciConnection conn) throws Exception{
		if(actions.size() == 0) {
			return;
		}
		
		ArrayList<SumoCommand> cmdList = new ArrayList<SumoCommand>();
		
		
		for(Map.Entry<Agent, AgentAction> entry: actions.entrySet()){
			
			Agent currentAgent = entry.getKey();
			SumoAgentAction action = (SumoAgentAction) entry.getValue();
			
			// TODO: remove this, changing color for debug purposes.
			if(action instanceof ChangeLaneAction){
				SumoCommand cmd = Vehicle.setColor(currentAgent.agentID, new SumoColor(255, 165, 0, 255));
				cmdList.add(cmd);
			}
			
			if(action == null)
				continue;
			cmdList.add(action.getCommand(currentAgent));
		}
		
	
		try {
			if(cmdList.size() != 0) {
				conn.do_jobs_set(cmdList);
			}
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

	@Override
	public StateData getNewStateData() {
		return getStateData(conn,true);
	}

	@Override
	public RoadNetwork updateRoadNetworkLanes(RoadNetwork rn){
		StateData stateData = getStateData(conn, false);

		HashMap<String,LaneData> laneDataMap = stateData.lanesData;
		for( Edge e : rn.getEdges()){
			Road r = e.getRoad();
			Lane l = r.getLanes()[0];
			LaneData laneData = laneDataMap.get(l.getID());
			r.setLength(laneData.length);
		}
		return rn;
	}

	/**
	 * Does a timestep in SUMO if 'timeStep' is true and returns StateData, which contains the most recent information about Edge, Lane and Agent objects.
	 * @param conn
	 * @param timeStep
	 * @return a StateData object which contains 3 HashMaps concerning agent, edge and lane data, and an Integer with the current timestep.
	 */
	public static StateData getStateData(SumoTraciConnection conn, boolean timeStep) {
		StateData stateData = null;
		QueryBuilder qb = new QueryBuilder();
		if(timeStep) {
			qb.doNextTimeStep();
		}
		qb.addQuerySubject(QuerySubject.Vehicle);
		qb.addQuerySubject(QuerySubject.Edge);
		qb.addQuerySubject(QuerySubject.Lane);
		qb.addQuerySubject(QuerySubject.Sensor);

		try {
			qb.addQueryField(QuerySubject.Vehicle, QueryField.Position);
			qb.addQueryField(QuerySubject.Vehicle, QueryField.Speed);
			qb.addQueryField(QuerySubject.Vehicle, QueryField.LeadingVehicle);
			qb.addQueryField(QuerySubject.Vehicle, QueryField.EdgeId);
			qb.addQueryField(QuerySubject.Vehicle, QueryField.LaneIndex);
	
			qb.addQueryField(QuerySubject.Edge, QueryField.MeanSpeed);
			qb.addQueryField(QuerySubject.Edge, QueryField.MeanTime);
			
			qb.addQueryField(QuerySubject.Lane, QueryField.MeanTime);
			qb.addQueryField(QuerySubject.Lane, QueryField.MeanSpeed);
			qb.addQueryField(QuerySubject.Lane, QueryField.EdgeId);
			qb.addQueryField(QuerySubject.Lane, QueryField.LaneLength);

			qb.addQueryField(QuerySubject.Sensor, QueryField.VehicleIDList);
			
			
			
			qb.executeQuery(conn);
			stateData = qb.getStateData();
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		return stateData;
	}
}
