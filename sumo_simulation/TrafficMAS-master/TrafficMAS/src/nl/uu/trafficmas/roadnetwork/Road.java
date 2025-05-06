package nl.uu.trafficmas.roadnetwork;

import java.util.List;

public class Road {
	public final String id;
	public double length;
	public final List<Lane> laneList;
	public int priority;
	private double meanTravelTime;
	private double meanSpeedEdge;
	public Road(String id, double length, List<Lane> laneList2, int priority) {
		this.id = id;
		this.length = length;
		this.laneList = laneList2;
		this.priority = priority;
		this.meanTravelTime = Double.MAX_VALUE;
	}
	
	public double getPriority(){
		return priority;
	}

	public void setLength(double length){
		this.length = length;
	}
	
	public void setMeanTravelTime(double meanTravelTime) {
		this.meanTravelTime = meanTravelTime;
	}
	
	public Lane[] getLanes() {
		Lane[] lanes = new Lane[laneList.size()];
		return laneList.toArray(lanes);
	}
	
	@Override
	public boolean equals(Object other) {
		if (other == null) return false;
		if (other == this) return true;
		if (!(other instanceof Road))return false;
		Road otherMyClass = (Road)other;
		
		
		return otherMyClass.id.equals(this.id);
	}

	public double getMeanTravelTime() {
		// TODO Auto-generated method stub
		return meanTravelTime;
	}
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("ID:"+id+"\r\n");
		sb.append("meanTime:"+meanTravelTime+"\r\n");

		for(Lane lane : laneList) {
			sb.append("\t LaneID:"+lane.getID()+"\r\n");
			sb.append("\t meanTimeLane:"+lane.getMeanTravelTime()+"\r\n");
		}
		return sb.toString();
		
	}

	public double getMeanSpeedEdge() {
		return meanSpeedEdge;
	}

	public void setMeanSpeedEdge(double meanSpeedEdge) {
		this.meanSpeedEdge = meanSpeedEdge;
	}
}
