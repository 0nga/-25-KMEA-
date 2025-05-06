package nl.uu.trafficmas.roadnetwork;

public class Edge {
	private Node from;
	private Node to;
	private Road road;
	
	public Edge(Node from, Node to, Road road) {
		this.from = from;
		this.to = to;
		this.road = road;
	}
	
	public Node getFromNode(){
		return from;
	}
	
	public Node getToNode(){
		return to;
	}
	
	public Road getRoad(){
		return road;
	}
	
	public String getID() {
		return road.id;
	}
	
	@Override 
	public String toString() {
		return from+" to "+to;
	}
	
	@Override
	public boolean equals(Object other) {
		if (other == null) return false;
		if (other == this) return true;
		if(!(other instanceof Edge))	return false;
		Edge castEdge = (Edge) other;
		
		if(!( this.getID().equals(castEdge.getID()))) return false;
		if(!this.getFromNode().equals(castEdge.getFromNode())) return false;
		if(!this.getToNode().equals(castEdge.getToNode())) return false;
		
		return this.getID().equals(castEdge.getID());
	}
	
	@Override
	public int hashCode() {
        return Integer.valueOf(this.getID().charAt(0));
    }
}
