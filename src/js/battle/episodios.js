//genereta battle episodes

var Episode = function Episode(props) {


    this.init = function(props, b, p){
        properties = props;
        battle = b;
        players = p;
        phase = "countdown";
        turn = 0;
        time = 0;


        				// Alternate CMP
		priorityAssignment = (priorityAssignment == 1) ? 0 : 1;
		for(var i = 0; i < players.length; i++){
			if(i == priorityAssignment){
				players[i].setPriority(1);
			} else{
				players[i].setPriority(0);
			}
		}

				// Set lead pokemon
				battle.setBattleMode("emulate");
				battle.setTurns(1);
				battle.setPlayers(players);
				battle.setNewPokemon(players[0].getTeam()[0], 0, false);
				battle.setNewPokemon(players[1].getTeam()[0], 1, false);
				battle.emulate(self.update);



}
}
