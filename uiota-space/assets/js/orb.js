(function(){
  const orb = document.querySelector(".orb");
  if(!orb) return;
  document.addEventListener("mousemove", e=>{
    const r = orb.getBoundingClientRect();
    const cx = r.left + r.width/2, cy = r.top + r.height/2;
    const dx = (e.clientX - cx)/r.width, dy = (e.clientY - cy)/r.height;
    orb.style.transform = `scale(1.03) translate(${dx*6}px, ${dy*6}px)`;
  });
  document.addEventListener("mouseleave", ()=>{
    orb.style.transform = "scale(1)";
  });
})();
