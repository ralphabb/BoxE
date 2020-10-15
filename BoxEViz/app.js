var rel_data = 0;
var ent_data = null;
var max_arity = 2;
var dataset = null;
var epoch_freq = 1;

var MAXIMUM_RANGE = 3;  // The range in which variables are plotted
var X_MARGIN = 0;
var Y_MARGIN = 0;
var width = 450;
var height = 450;
var checks_per_line = 5;

var svg = d3.select("body").append("svg").attr("height", height).attr("width", width);  // Canvas
var linScaleX = d3.scaleLinear().domain([-MAXIMUM_RANGE, MAXIMUM_RANGE]).range([X_MARGIN,X_MARGIN + width]);
var linScaleY = d3.scaleLinear().domain([MAXIMUM_RANGE, -MAXIMUM_RANGE]).range([Y_MARGIN,Y_MARGIN + height]);


function move_boxes(boxes, time_step, anim_time, pos, offset){  // Modular Transition Mechanism
    boxes.transition().duration(anim_time).delay(time_step * anim_time - offset) // Box Animation
                    .attr("x", function (d, ix){return linScaleX(d["x"][time_step][pos]);})
                    .attr("y", function (d, ix){return linScaleY(d["y"][time_step][pos]);})
                    .attr("width", function (d, ix){return (d["width"][time_step][pos]/(2 * MAXIMUM_RANGE))*width;})
                    .attr("height", function (d, ix){return (d["height"][time_step][pos]/(2 * MAXIMUM_RANGE))*height;})
                    .on("end", function(){document.querySelector('input[id=animSlider]').value = time_step;
                                          d3.select("#animSlider").attr("value", time_step);
                                          d3.select("#epochText").text("Epoch "+time_step);});
    return boxes;
}

function compute_bumps(ent_data, ent_idx, time_step){
    base_x = ent_data[ent_idx]["x"][time_step];
    base_y = ent_data[ent_idx]["y"][time_step];
    bumped_x = []
    bumped_y = []
    for (idx in ent_data){
        bumping_ent = ent_data[idx];
        b_x = base_x + bumping_ent["b_x"][time_step];
        b_y = base_y + bumping_ent["b_y"][time_step];
        bumped_x.push(b_x);
        bumped_y.push(b_y);
     }
    return {x: bumped_x, y: bumped_y};
}

function move_ents(t_step, anim_time, offset){
base_points = d3.selectAll("circle").transition()
                .duration(anim_time).delay(t_step * anim_time - offset)
                .attr("cx", function(d, ix){return linScaleX(d["x"][t_step]);})
                .attr("cy", function(d, ix){return linScaleY(d["y"][t_step]);});
    // Now Draw the bumped points
    for (i=0; i<ent_data.length;i++){
        bumps = compute_bumps(ent_data, i, t_step);
        for (j=0;j<ent_data.length;j++){
            d3.select(".entbump"+i).select(".bump"+j).transition()
                .duration(anim_time).delay(t_step * anim_time - offset)
                .attr("x1",linScaleX(ent_data[i]["x"][t_step]))
                .attr("y1",linScaleY(ent_data[i]["y"][t_step]))
                .attr("x2",linScaleX(bumps.x[j]))
                .attr("y2",linScaleY(bumps.y[j]));

            }
    }

}

function move_texts(texts, time_step, anim_time, pos, offset){  // Modular Transition Mechanism
    texts.transition().duration(anim_time).delay(time_step * anim_time - offset) // Box Animation
                    .attr("x", function (d, ix){return linScaleX(d["x"][time_step][pos]);})
                    .attr("y", function (d, ix){return linScaleY(d["y"][time_step][pos])-7;})
    return texts;
}

var colorScale = d3.scaleOrdinal(d3.schemeCategory10);

// Step 1: Load Dataset
var data = d3.json('data.json').then(function (data){
    rel_data = data["Relations"];
    ent_data = data["Entities"];
    max_arity = data["Max Arity"];
    dataset = data["Dataset"];
    epoch_freq = data["EpochFrequency"];
    anim_time = data["animTime"];
    nb_stages = data["nbStages"];
    d3.select("body").select("h3").text("Dataset: "+dataset);
    // Step 2: Prepare intermediate markers
    for (i=0; i < ent_data.length; i++){
        svg.append("defs").append("svg:marker").attr("id", "arrow"+i).attr("refX", 2).attr("refY", 6)
	.attr("markerWidth", 13).attr("markerHeight", 13).attr("orient", "auto").append("svg:path")
	.attr("d", "M2,2 L2,11 L10,6 L2,2").attr("fill", colorScale(i)).attr("fill-opacity", 0.85);
    }
    // Step 3: Set up the slider and button
    var animContainer = d3.select("#animController").style("height", "80px");
    var slider = d3.select("#animSlider").attr("max", nb_stages - 1).attr("value", 0).style("position", "absolute")
                                        .style("width", width - 80+"px").style("left", "80px");
    var playPause = d3.select("#playPause").style("position", "absolute").style("width", 60+"px");
    var epochText = d3.select("#epochText").style("position", "absolute").style("width", width - 80+"px")
                        .style("left", "80px");
    // Step 4: Set up boxes and entity representations, within the data loading promise
    var groups = svg.selectAll("g.boxrel").data(rel_data).enter().append("g")
                                             .attr("class", function(d,i){return "boxrel"+i}) // Box Group

    var ent_groups = svg.selectAll("g.entbump").data(ent_data).enter().append("g")
                                               .attr("class", function(d,i){return "entbump"+i});

    for (i=1; i <= max_arity;i++){ // Add a Box Per Arity
        var boxes = groups.append("rect").attr("class", function (d, ix){return "rel"+ix;})
                             .classed("pos"+i, true)  // Initial Box Location
                             .attr("x", function (d, ix){return linScaleX(d["x"][0][i-1]);})
                             .attr("y", function (d, ix){return linScaleY(d["y"][0][i-1]);})
                             .attr("width", function (d, ix){return (d["width"][0][i-1]/(2 * MAXIMUM_RANGE))*width;})
                             .attr("height", function (d, ix){return (d["height"][0][i-1]/(2 * MAXIMUM_RANGE))*height;})
                             // Click Response
                             .on("click", function(d, ix){
                             current_elt = d3.select(this);
                             if (current_elt.classed("selected")){
                                 d3.selectAll("*[class^=boxrel]").style("visibility", "visible").classed("selected", false);
                                 d3.selectAll("*[class^=boxrel] > text").classed("selected", false);
                                 d3.selectAll("*[class^=boxrel] > rect").classed("selected", false);
                                 // Logic for check boxes
                                 d3.selectAll("*[name^=rel]").property("checked", true);
                                } else {
                                 d3.selectAll("*[class^=boxrel]").style("visibility", "hidden").classed("selected", false);
                                 d3.selectAll("*[class^=boxrel] > text").classed("selected", false);
                                 d3.selectAll("*[class^=boxrel] > rect").classed("selected", false);
                                 d3.selectAll(".boxrel"+ix).style("visibility", "visible").classed("selected", true);
                                 d3.selectAll(".boxrel"+ix + "> text").classed("selected", true);
                                 d3.selectAll(".boxrel"+ix + "> rect").classed("selected", true);
                                 // Logic for check boxes
                                 d3.selectAll("*[name^=rel]").property("checked", false);
                                 d3.select("*[name=rel"+ix).property("checked", true);
                                }
                             });

         var texts = groups.append("text").attr("class", function (d, ix){return "rel"+ix;})
                                           .classed("pos"+i, true)
                                           .attr("x", function (d, ix){return linScaleX(d["x"][0][i-1]);})
                                           .attr("y", function (d, ix){return linScaleY(d["y"][0][i-1]) - 10;})
                                           .text(function (d) { return d.name + ": Pos "+i; })
                                           .attr("font-family", "Geneva")
                                           .attr("font-size", "12px")
                                           .attr("fill", "black");

        for (time_step=1; time_step < nb_stages; time_step++){// Animate
                 boxes = move_boxes(boxes, time_step, anim_time, i-1, 0);
                 texts = move_texts(texts, time_step, anim_time, i-1, 0);
             }

     };

    // Step 5: Entity Representations:
    t_step = 0;
    var base_points = ent_groups.append("circle").attr("r","4px").attr("cx", function(d, ix){return linScaleX(d["x"][t_step]);})
                                                                 .attr("cy", function(d, ix){return linScaleY(d["y"][t_step]);})
                                                                 .attr("fill", function(d, ix){return colorScale(ix);})
                                                                 .attr("class", function(d, ix){return "base"+ix;})
                                                                 .attr("z-index", "100");
    // Now Draw the bumped points
    for (i=0; i<ent_data.length;i++){
        bumps = compute_bumps(ent_data, i, t_step);
        for (j=0;j<ent_data.length;j++){
            d3.select(".entbump"+i).append("line").attr("x1",linScaleX(ent_data[i]["x"][t_step]))
                                     .attr("y1",linScaleY(ent_data[i]["y"][t_step]))
                                     .attr("x2",linScaleX(bumps.x[j]))
                                     .attr("y2",linScaleY(bumps.y[j]))
                                     .attr("marker-end", "url(#arrow"+j+")")
                                     .attr("stroke", colorScale(j))
                                     .classed("base"+i, true)
                                     .classed("bump"+j, true);


            }
    }

     for (time_step=1; time_step < nb_stages; time_step++){
        move_ents(time_step, 100,0);
     }

    // Step 6: Set Up Slider Dynamics:
    slider = slider.on("input", function(){
    d3.selectAll('*').transition(); // Stop any running animations
    playPause.text("Play")
    numeric_val = Number(this.value)
    d3.select("#epochText").text("Epoch "+numeric_val * epoch_freq); // To not wait for the end of the animation
    for (i=1; i <= max_arity;i++){
        var boxes = d3.selectAll('*[class^=boxrel] > rect.pos'+i);   // Select all boxes on screen
        var texts = d3.selectAll('*[class^=boxrel] > text.pos'+i); // And texts
        move_boxes(boxes, numeric_val, anim_time, i-1, numeric_val * anim_time);
        move_texts(texts, numeric_val, anim_time, i-1, numeric_val * anim_time);
        slider.attr("value", numeric_val)  // Set new value for the slider
        }
    move_ents(numeric_val, anim_time, numeric_val * anim_time);
    });


    // Step 7: Play/Pause button events
    playPause.on("click", function(){
    var button = d3.select(this)
    if (button.text() == "Pause"){
        d3.selectAll('*').transition();
        button.text("Play");
        } else { // Was Paused,
        time_step_start = Math.round(slider.attr("value"));
         button.text("Pause");
         for (i=1; i <= max_arity;i++){
            var boxes = d3.selectAll('*[class^=boxrel] > rect.pos'+i); // Select all boxes on screen
            var texts = d3.selectAll('*[class^=boxrel] > text.pos'+i); // And texts
            for (time_step=time_step_start; time_step < nb_stages; time_step++){
                 boxes = move_boxes(boxes, time_step, anim_time, i-1, time_step_start * anim_time);
                 texts = move_texts(texts, time_step, anim_time, i-1, time_step_start * anim_time);
                 }
            }
         for (time_step=time_step_start; time_step < nb_stages; time_step++){
            move_ents(time_step, anim_time, time_step_start * anim_time);
         }
       }
    });
    // Step 8: Set up visual section to make boxes visible / invisible
    var controlContainer = d3.select("#relController").style("position", "relative").style("height", 30*Math.ceil(rel_data.length / checks_per_line)+"px");
    controlContainer.selectAll("input").data(rel_data).enter().append("input").attr("checked", true)
                                       .attr("type", "checkbox").attr("name", function (d, ix){return "rel"+ix})
                                       .style("position", "absolute")
                                       .style("left", function(d,ix){return 120*(ix % checks_per_line)+"px";})
                                       .style("top", function(d,ix){return 30*Math.floor(ix / checks_per_line)+"px";})
                                       .on("click", function(d, ix){
                                           checkb = d3.select(this);
                                           if (!checkb.property("checked")){ // If checked
                                                d3.select('.boxrel'+ix).style("visibility", "hidden").classed("selected", false);
                                           } else {
                                                d3.select('.boxrel'+ix).style("visibility", "visible").classed("selected", true);
                                           }
                                       });
    // Labels
    controlContainer.selectAll("label").data(rel_data).enter().append("label").attr("for", function (d, i){return "rel"+i})
                                       .text(function(d){ return d["name"];}).style("position", "absolute")
                                       .style("left", function(d,ix){return 20 + 120*(ix % checks_per_line)+"px";})
                                       .style("top", function(d,ix){return 30*Math.floor(ix / checks_per_line)+"px";})
                                       .style("overflow", "scroll")
                                       .style("width", "90px");

    // Step 9: Set up visual section to make boxes visible / invisible
    var entContainer = d3.select("#entController").style("position", "relative").style("height", 30*Math.ceil(ent_data.length / checks_per_line)+"px");
    entContainer.selectAll("input").data(ent_data).enter().append("input").attr("checked", true)
                                       .attr("type", "checkbox").attr("name", function (d, ix){return "ent"+ix})
                                       .style("position", "absolute")
                                       .style("left", function(d,ix){return 120*(ix % checks_per_line)+"px";})
                                       .style("top", function(d,ix){return 30*Math.floor(ix / checks_per_line)+"px";})
                                       .on("click", function(d, ix){
                                           checkb = d3.select(this);
                                           if (!checkb.property("checked")){ // If now checked
                                                d3.select('.entbump'+ix).style("visibility", "hidden");
                                                d3.selectAll('.bump'+ix).style("visibility", "hidden");
                                           } else {
                                                d3.select('.entbump'+ix).style("visibility", "visible");
                                                d3.selectAll('.bump'+ix).style("visibility", null);
                                           }
                                       });
    // Labels
    entContainer.selectAll("label").data(ent_data).enter().append("label").attr("for", function (d, i){return "ent"+i})
                                       .text(function(d){ return d["name"];}).style("position", "absolute")
                                       .style("left", function(d,ix){return 20 + 120*(ix % checks_per_line)+"px";})
                                       .style("top", function(d,ix){return 30*Math.floor(ix / checks_per_line)+"px";})
                                       .style("overflow", "scroll")
                                       .style("width", "90px");
});




