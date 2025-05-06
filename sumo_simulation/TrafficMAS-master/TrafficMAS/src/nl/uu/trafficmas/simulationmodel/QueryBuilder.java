package nl.uu.trafficmas.simulationmodel;

import it.polito.appeal.traci.SumoTraciConnection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;

import de.tudresden.sumo.cmd.Simulation;
import de.tudresden.sumo.config.Constants;
import de.tudresden.sumo.util.SumoCommand;
import de.tudresden.ws.container.SumoStringList;

public class QueryBuilder {
	private LinkedHashSet<QuerySubject> 	querySubjects;
	private LinkedHashMap<QuerySubject,
		LinkedHashSet<QueryField>> 			querySubjectFields;
	private LinkedHashMap<QuerySubject,
		List<String>> 						subjectIDs;
	
	private StateData 							stateData;
	private HashMap<String,AgentData> 			agentsData;
	private HashMap<String,EdgeData> 			edgesData;
	private HashMap<String,LaneData> 			lanesData;
	private LinkedHashMap<String, SensorData> 	sensorData;

	private boolean timeStep;
	
	public QueryBuilder() {
		querySubjects 		= new LinkedHashSet<QuerySubject>();
		querySubjectFields 	= new LinkedHashMap<QuerySubject,LinkedHashSet<QueryField>>();
		subjectIDs			= new LinkedHashMap<QuerySubject,List<String>>();
		timeStep 			= false;
	}
	
	public StateData getStateData() {
		return stateData;
	}

	public HashMap<String, AgentData> getAgentsData() {
		return agentsData;
	}

	public HashMap<String, EdgeData> getEdgesData() {
		return edgesData;
	}

	public HashMap<String, LaneData> getLanesData() {
		return lanesData;
	}

	public void addQuerySubject(QuerySubject sq) {
		querySubjects.add(sq);
		if(!querySubjectFields.containsKey(sq)){
			querySubjectFields.put(sq, new LinkedHashSet<QueryField>());
		}
	}
	
	public void addQueryField(QuerySubject sq, QueryField sf) throws Exception {
		if(!querySubjects.contains(sq)) {
			this.addQuerySubject(sq);
		}
		querySubjectFields.get(sq).add(sf);
	}
	
	public void executeQuery(SumoTraciConnection conn) throws Exception {
		// for what subjects do we need to get info?
		List<SumoCommand> cmdList = new ArrayList<SumoCommand>();
		if(timeStep) {
			cmdList.add(new SumoCommand(Constants.CMD_SIMSTEP2, 0));

		}
		cmdList.add(de.tudresden.sumo.cmd.Simulation.getCurrentTime());
		cmdList.add(Simulation.getDepartedIDList());

		for(QuerySubject querySubject : querySubjects) {
			cmdList.add(querySubject.getIDListCommand());
		}
		LinkedList<Object> responses = conn.do_jobs_get(cmdList);
		cmdList.clear();
		
		// first response is null/or subscribed events
		if(timeStep) {
			responses.remove(0);
		}
		int currentTimeStep = (int) responses.remove(0);
		SumoStringList departedIDListSumo = (SumoStringList) responses.remove(0);
		String [] list = new String[departedIDListSumo.size()];
		
		departedIDListSumo.toArray(list);
		List<String> departedIDList = Arrays.asList(list);
		// generate the new queries
		generateQueries(cmdList, responses);
		// responses should now be empty!
		assert(responses.isEmpty());

		
		// get new responses
		responses = conn.do_jobs_get(cmdList);
		processResponses(responses);
		assert(responses.isEmpty());
		
		stateData = new StateData(agentsData, edgesData, lanesData, sensorData, currentTimeStep, departedIDList);
		
		querySubjects.clear();
	}

	private void processResponses(LinkedList<Object> responses) {
		agentsData 	= new LinkedHashMap<>();
		edgesData 	= new LinkedHashMap<>();
		lanesData 	= new LinkedHashMap<>();
		sensorData	= new LinkedHashMap<>();
		for(QuerySubject querySubject : querySubjects) {
			int numberOfFields = querySubjectFields.get(querySubject).size();
			for(String id : subjectIDs.get(querySubject)) {
				List<Object> subList = responses.subList(0, numberOfFields);
				
				Data data = querySubject.toData(querySubjectFields.get(querySubject), subList, id);
				
				switch(querySubject) {
				case Edge:
					edgesData.put(id, (EdgeData) data);
					break;
				case Lane:
					lanesData.put(id, (LaneData) data);
					break;
				case Vehicle:
					agentsData.put(id, (AgentData) data);
					break;
				case Sensor:
					sensorData.put(id, (SensorData) data);
					break;
				}
				
				responses.subList(0, numberOfFields).clear();
			}
		}
	}

	private void generateQueries(List<SumoCommand> cmdList,
			LinkedList<Object> responses) throws Exception {
		// how many ids are there for each subject,
		// and what field do we need to retrieve for each id?
		for(QuerySubject querySubject : querySubjects) {
			//SumoStringList s				= (SumoStringList) responses.remove(0);
			SumoStringList subList 	= (SumoStringList) responses.remove();
			List<String> idList = new ArrayList<String>(subList);
			subjectIDs.put(querySubject, idList);
			// prepare queries
			for(String id : subjectIDs.get(querySubject)) {
				for(QueryField queryField : querySubjectFields.get(querySubject)) {
					cmdList.add(querySubject.queryFieldToCommand(queryField,id));
				}
			}
		}
	}

	public void doNextTimeStep() {
		// TODO Auto-generated method stub
		this.timeStep = true;
	}
}
