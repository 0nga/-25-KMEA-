package nl.uu.trafficmas.organisation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import nl.uu.trafficmas.norm.NormInstantiation;
import nl.uu.trafficmas.norm.NormScheme;
import nl.uu.trafficmas.norm.Sanction;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Sensor;
import nl.uu.trafficmas.simulationmodel.AgentData;

public class Organisation implements Observer{
	public final String 						id;
	private List<NormScheme> 					normSchemes;
	private List<NormInstantiation> 			normInstantiations;
	private Map<String,Set<NormInstantiation>> 	agentNormInst;
	private Map<String,List<Sanction>> 			agentSanctions;
	private List<Sensor> 						sensors;
	private Map<String,AgentData> 				currentOrgKnowledge;
	private List<InstitutionalState> 			institutionalStates;
	private int 								currentTime;
	
	public Organisation(String id, List<NormScheme> normSchemes, List<Sensor> sensors ){
		this.id 			= id;
		this.normSchemes 	= normSchemes;
		this.sensors 		= sensors;
		normInstantiations	= new ArrayList<NormInstantiation>();
		agentNormInst		= new HashMap<String, Set<NormInstantiation>>();
		agentSanctions		= new HashMap<String, List<Sanction>>();
	}
	
	public List<NormScheme> getNormSchemes() {
		return normSchemes;
	}
	public void setNormSchemes(ArrayList<NormScheme> normSchemes) {
		this.normSchemes = normSchemes;
	}
	
	public List<NormInstantiation> getNormInstantiations() {
		return normInstantiations;
	}
	
	public List<Sensor> getSensors() {
		return sensors;
	}
	public void setSensors(ArrayList<Sensor> sensors) {
		this.sensors = sensors;
	}
	
	public List<InstitutionalState> getInstitutionalStates() {
		return institutionalStates;
	}
	public void setInstitutionalStates(List<InstitutionalState> institutionalStates) {
		this.institutionalStates = institutionalStates;
	}
	public List<Sanction> getNewSanctions() {
		List<Sanction> sanctions = new ArrayList<Sanction>();
		for(NormInstantiation ni : normInstantiations) {
			AgentData ad = currentOrgKnowledge.get(ni.agentID());
			
			if(ad != null && ni.violated(ad)) {
				Sanction sanction = ni.getSanction(ad.id);
				sanctions.add(sanction);
				if(!agentSanctions.containsKey(ad.id)) {
					agentSanctions.put(ad.id, new ArrayList<Sanction>());
				}
				agentSanctions.get(ad.id).add(sanction);
				// remove instantiation when sanctioned.
				// TODO: also get this to agent
				//agentNormInst.get(ni.agentID).remove(ni);
			}
		}
		
		return sanctions;
	}
	
	public List<NormInstantiation> getNewNormInstantiations(RoadNetwork rn) {
		return getNewNormInstantiations(rn, normSchemes, currentOrgKnowledge, agentNormInst, normInstantiations, currentTime);
	}
	
	public static List<NormInstantiation> getNewNormInstantiations(
			RoadNetwork rn, List<NormScheme> normSchemes,
			Map<String, AgentData> currentOrgKnowledge,
			Map<String,Set<NormInstantiation>> agentNormInst,
			List<NormInstantiation> normInstantiations,
			int currentTime) {
		List<NormInstantiation> newList = new ArrayList<NormInstantiation>();
		
		for(NormScheme ns : normSchemes) {
			if(ns.checkCondition(currentOrgKnowledge)) {
				List<NormInstantiation> nis = ns.instantiateNorms(rn,currentTime, currentOrgKnowledge);
				newList.addAll(nis);
				for(NormInstantiation ni : nis) {
					if(!agentNormInst.containsKey(ni.agentID)) {
						agentNormInst.put(ni.agentID, new HashSet<NormInstantiation>());
					}
					agentNormInst.get(ni.agentID).add(ni);
				}
			}
		}
		normInstantiations.addAll(newList);
		return newList;
	}
	public List<NormInstantiation> getClearedNormInstantiations() {
		List<NormInstantiation> clearedList= new ArrayList<NormInstantiation>();
		for(NormInstantiation ni : normInstantiations) {
			//TODO: also delete on norm violation?
			if(ni.deadline(currentOrgKnowledge, currentTime)) {
				clearedList.add(ni);
			}
		}
		for(NormInstantiation ni : clearedList) {
			agentNormInst.get(ni.agentID).remove(ni);
		}
		normInstantiations.removeAll(clearedList);
		return clearedList;
	}
	
	public void readSensors() {
		currentOrgKnowledge = new HashMap<String,AgentData>();
		for(Sensor s : sensors) {
			for(AgentData ad : s.readSensor()) {
				// TODO: proper error handling when an agent is detected by *two* sensors
				if(currentOrgKnowledge.containsKey(ad.id)) {
					System.out.println("Org.readsensors: agent " + ad.id + " was already located by a sensor!");
					AgentData oldAD = currentOrgKnowledge.get(ad.id);
					System.out.println("Its old road:"+ oldAD.roadID + ", lane:"+ (oldAD.laneIndex) + " and pos:" + oldAD.position);
					System.out.println("Its new road:"+ ad.roadID + ", lane:"+ (ad.laneIndex) + " and pos:" + ad.position);
				}
				currentOrgKnowledge.put(ad.id, ad);
			}
		}
	}
	public void updateTime(int currentTime) {
		this.currentTime = currentTime;
	}

	public Map<String,AgentData> getKnowledge() {
		return currentOrgKnowledge;
	}

	@Override
	public void receiveInformation(Object o) {
		if(o instanceof Map) {
			Set<Entry<String, AgentData>> entrySet = ((Map<String,AgentData>) o).entrySet();
			for(Entry<String, AgentData> e : entrySet) {
				currentOrgKnowledge.put(e.getKey(), e.getValue());
			}
		}
	}
}
