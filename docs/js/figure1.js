// Create SVG containers
var figure1 = d3.select("figure1");

var margin = {top: 30, right: 30, bottom: 60, left: 60},
    width = 460 - margin.left - margin.right,
    height = 300 - margin.top - margin.bottom;

var X_MAX = 10;
var Y_MAX = 10;

var Y_BAR_INIT = 2;

// append the svg object to the body of the page
var figure1 = d3.select("#figure1");


var svgLeft = figure1.append("svg").attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")");

// Define scales
var xScale = d3.scaleLinear()
    .domain([0, X_MAX])  // your data range
    .range([0, width]);  // your pixel range

var yScale = d3.scaleLinear()
    .domain([0, Y_MAX])  // your data range
    .range([height, 0]);  // your pixel range

// Create axes
var xAxis = d3.axisBottom(xScale);
var yAxis = d3.axisLeft(yScale).ticks(5);

// Append axes to SVG
svgLeft.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(xAxis);

svgLeft.append("g")
    .call(yAxis);

function uniform(low, high) {
    return Math.random() * (high - low) + low;
}

function searchsorted(array, value, side = "left") {
    if (side === "left") {
        return array.findIndex(el => value < el);
    } else {
        return array.findIndex(el => value <= el);
    }
}

function gaussianMixtureSample(locs, scales) {
    // Randomly select a component
    var component = Math.floor(Math.random() * locs.length);

    // Get the parameters of the selected component
    var loc = locs[component];
    var scale = scales[component];

    // Generate a sample from the selected component
    var sample = normalSample(loc, scale);

    return sample;
}

function normalSample(loc, scale) {
    // Box-Muller transform
    var u1 = Math.random();
    var u2 = Math.random();
    var z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    var sample = z0 * scale + loc;

    return sample;
}

function normalDensity(x, loc, scale) {
    var a = 1 / (scale * Math.sqrt(2 * Math.PI));
    var b = Math.exp(-0.5 * Math.pow((x - loc) / scale, 2));
    return a * b;
}

// Generate scatter plot data
function branchingMixtureSample(x_splits = [0, 0.33, 0.66], scale = 0.075) {
    let x = uniform(0, 1);
    const idx = searchsorted(x_splits, x, "right");
    const locs = [x].concat(x_splits.slice(0, idx).map(split => 2 * split - x));
    const scales = new Array(locs.length).fill(scale);
    let y = gaussianMixtureSample(locs, scales);
    y = y * 4 + 5
    x = x * 10;
    return [x, y];
}


function branchingMixtureDensity(y, x, x_splits = [0, 0.33, 0.66], scale = 0.075) {
    x = x / 10;
    y = (y - 5) / 4;
    const idx = searchsorted(x_splits, x, "right");
    const locs = [x].concat(x_splits.slice(0, idx).map(split => 2 * split - x));
    const scales = new Array(locs.length).fill(scale);
    let densities = locs.map((loc, i) => normalDensity(y, loc, scales[i]));
    return d3.mean(densities);
}


// Scatter some dots in the left SVG
var dots = svgLeft.selectAll("circle").data(d3.range(300).map(function () {
    // sample branchingMixtureSample
    let [x, y] = branchingMixtureSample();
    const swap = x;
    x = y;
    y = swap
    return {x: x, y: y};
}));


dots.enter().append("circle").attr("r", 2.5).attr("cx", function (d) {
    return xScale(d.x);
}).attr("cy", function (d) {
    return yScale(d.y);
}).attr("fill", "orange")
    .attr("stroke", "black")
    .attr("stroke-width", 0.3);

// Add a horizontal line (slider) that can be dragged up and down
var drag = d3.drag().on("drag", function () {
    var y = Math.max(0, Math.min(Y_MAX, yScale.invert(d3.event.y)));
    d3.select(this).attr("y1", yScale(y)).attr("y2", yScale(y));
    d3.select("#sliderText").attr("y", yScale(y) - 5).text("x = " + y.toFixed(2));
    updateGaussian(y);  // Update the Gaussian density plot when the slider is moved
    // Update the y-axis label of the density plot
    d3.select("#yAxisLabelDensity").text("p(y | x = " + y.toFixed(2) + " )");
});
var slider = svgLeft.append("line").attr("x1", 0).attr("x2", width).attr("y1", yScale(Y_BAR_INIT)).attr("y2", yScale(Y_BAR_INIT))
    .attr("stroke", "black")
    .attr("stroke-width", 2).attr("class", "draggable").call(drag);

svgLeft.append("text")
    .attr("id", "sliderText")
    .attr("x", width) // position the text at the right end of the line
    .attr("y", yScale(Y_BAR_INIT) - 5) // position the text at the same height as the line
    .attr("text-anchor", "end") // right align the text
    .text("x = " + Y_BAR_INIT.toFixed(2)); // set the text content


svgLeft.append("text")
    .attr("transform", "translate(" + (width/2) + " ," + (height + margin.top + 10) + ")")
    .style("text-anchor", "middle")
    .style("font-family", "Times New Roman")
    .text("y");

// Add Y Axis label
svgLeft.append("text")
    .attr("transform", "rotate(-90)")
    .attr("y", 0 - margin.left + 10)
    .attr("x",0 - (height / 2))
    .attr("dy", "1em")
    .style("text-anchor", "middle")
    .text("x");


// Now do the density plot on the right

var svgRight = figure1.append("svg").attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")");


// Define scales
var xScaleRight = d3.scaleLinear()
    .domain([0, Y_MAX])  // your data range
    .range([0, width]);  // your pixel range

var yScaleRight = d3.scaleLinear()
    .domain([0, 6])  // your data range
    .range([height, 0]);  // your pixel range

// Create axes
var xAxis = d3.axisBottom(xScaleRight);
var yAxis = d3.axisLeft(yScaleRight).ticks(4);

// Append axes to SVG
svgRight.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(xAxis);

svgRight.append("g")
    .call(yAxis);

svgRight.append("text")
    .attr("transform", "translate(" + (width/2) + " ," + (height + margin.top + 10) + ")")
    .style("text-anchor", "middle")
    .style("font-family", "Times New Roman")
    .text("y");

// Add Y Axis label
svgRight.append("text")
    .attr("transform", "rotate(-90)")
    .attr("y", 0 - margin.left + 10)
    .attr("x",0 - (height / 2))
    .attr("dy", "1em")
    .style("text-anchor", "middle")
    .attr("id", "yAxisLabelDensity")
    .text("p(y | x = " + Y_BAR_INIT.toFixed(2) + " )");


// Function to update the Gaussian density plot
function updateGaussian(x_conditioned) {
    // draw gaussian density from x = 0 to x = 10 with mean y and std 1
    var x = d3.range(0, 10, 0.01);
    var y = x.map(function (y) {
        return branchingMixtureDensity(y, x_conditioned);
    });

    var line = d3.line()
        .x(function (d, i) {
            return xScaleRight(x[i]);
        })
        .y(function (d) {
            return yScaleRight(d);
        });

   d3.select("#densityPath")
        .datum(y)
        .attr("d", line);

}


// draw gaussian density from x = 0 to x = 10 with mean 5 and std 1
var x = d3.range(0, 10, 0.01);
var y = x.map(function (y) {
    return branchingMixtureDensity(y, Y_BAR_INIT);
});

var line = d3.line()
    .x(function (d, i) {
        return xScaleRight(x[i]);
    })
    .y(function (d) {
        return yScaleRight(d);
    });

svgRight.append("path")
    .attr("id", "densityPath")
    .datum(y)
    .attr("fill", "none")
    .attr("stroke", "#266E15FF")
    .attr("stroke-width", 3)
    .attr("d", line);
