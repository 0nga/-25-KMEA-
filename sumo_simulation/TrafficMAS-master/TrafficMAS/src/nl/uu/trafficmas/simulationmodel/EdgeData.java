package nl.uu.trafficmas.simulationmodel;

public class EdgeData implements Data {

	public final String id;
	public final double meanSpeed;
	public final double meanTime;

	public EdgeData(String id , double meanSpeed, double meanTime) {
		this.id = id;
		this.meanSpeed = meanSpeed;
		this.meanTime = meanTime;
	}

}
