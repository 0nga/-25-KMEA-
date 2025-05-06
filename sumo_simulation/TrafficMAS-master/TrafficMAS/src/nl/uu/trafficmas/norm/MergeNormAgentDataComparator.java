package nl.uu.trafficmas.norm;

import java.util.Comparator;

import nl.uu.trafficmas.simulationmodel.AgentData;

public class MergeNormAgentDataComparator implements Comparator<AgentData> {
    @Override
    public int compare(AgentData o1, AgentData o2) {
        return (o1.position > o2.position) ? -1 : 1;
    }
}