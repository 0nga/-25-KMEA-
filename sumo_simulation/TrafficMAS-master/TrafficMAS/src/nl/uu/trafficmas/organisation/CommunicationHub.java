package nl.uu.trafficmas.organisation;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class CommunicationHub<Observee> {
	
	Map<Observee,Set<Observer>> subscriptions;
	
	
	public CommunicationHub() {
		subscriptions = new HashMap<Observee,Set<Observer>>();
	}
	
	public void addSubscription(Observee o1 ,Observer o2 ) {
		if(!subscriptions.containsKey(o1)) {
			subscriptions.put(o1, new HashSet<Observer>());
		}
		
		subscriptions.get(o1).add(o2);
	}
	
	public void shareInformation(Observee o1, Object object) {

		if(subscriptions.get(o1) == null) 
			return;
		for(Observer o : subscriptions.get(o1)) {
			o.receiveInformation(object);
		}
	}
	
}
